import os 
import json
import shutil
import torch
import torch.nn as nn
import pandas as pd 
import numpy as np
import time
import tqdm

from typing import Any
from typing import *
from pathlib import Path as path
from transformers import set_seed

from utils import catch_and_record_error
from arguments import CustomArgs
from logger import CustomLogger, LOG_FILENAME_DICT
from data import CustomData, DataLoader
from model import get_model
from model.configs import *
from metrics import ComputeMetrics
from analyze import analyze_metrics_json


class Trainer:
    def __init__(self) -> None:
        self.device = None
        pass
    
    def fit(
        self,
        args:CustomArgs,
        model:nn.Module,
        data:CustomData, 
        compute_metrics:ComputeMetrics,
        logger:CustomLogger,
    ):
        self.device = torch.device('cuda:0') if args.cuda_id else torch.device('cpu')
        
        data.prepare_dataloader(train_batch_size=args.train_batch_size, 
                                eval_batch_size=args.eval_batch_size)
        train_batch_num = len(data.train_dataloader)
        total_batch_num = train_batch_num*args.epochs
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, total_iters=total_batch_num
        )
        
        model.to(self.device)
        model.train()
        
        cur_batch_num = 0
        total_loss, log_loss = 0, 0
        best_metrics = {m:float('inf') for m in compute_metrics.metric_names}
        best_model_file = path(args.ckpt_dir, 'best.bin')
        progress_bar = tqdm.tqdm(desc=f'train', total=total_batch_num)
        
        for _ in range(args.epochs):
            for inputs in data.train_dataloader:
                cur_batch_num += 1
                
                inputs = inputs.to(self.device)
                output = model(inputs)
                loss:torch.Tensor = output['loss']
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update()
                
                total_loss += loss
                log_loss += loss
                if not cur_batch_num % args.log_steps:
                    cur_log = {
                        'loss': float((log_loss/args.log_steps).cpu()),
                        'lr': lr_scheduler.get_last_lr()[0],
                        'epoch': cur_batch_num/train_batch_num,
                    }
                    log_loss = 0
                    logger.log_json(cur_log, LOG_FILENAME_DICT['loss'], log_info=False, mode='a')
                if not cur_batch_num % args.eval_steps:
                    progress_bar.display()
                    metrics = self.evaluate(model, data.dev_dataloader, compute_metrics)
                    if metrics['MSE'] < best_metrics['MSE']:
                        torch.save(model, best_model_file)
                    for k,v in metrics.items():
                        best_metrics[k] = min(best_metrics[k], v)
                    logger.log_json(
                        logger.add_prefix_string(metrics, 'dev_'),
                        LOG_FILENAME_DICT['dev'], log_info=False, mode='a')
                    logger.log_json(
                        logger.add_prefix_string(best_metrics, 'best_'),
                        LOG_FILENAME_DICT['best'], log_info=False, mode='w')
        
        progress_bar.close()
        model = torch.load(best_model_file)
        test_metric = self.evaluate(model, data.test_dataloader, compute_metrics)
        logger.log_json(
            logger.add_prefix_string(test_metric, 'test_'),
            LOG_FILENAME_DICT['test'], log_info=True, mode='w')
        
        return {'train_loss': float((total_loss/cur_batch_num).cpu()), 
                'train_epoch':cur_batch_num/train_batch_num}

    def evaluate(
        self,
        model:nn.Module,
        dataloader:DataLoader,
        compute_metrics:ComputeMetrics,
    ):
        model.eval()
        with torch.no_grad():
            pred, gt = [], []
            for inputs in tqdm.tqdm(dataloader, desc='evaluate'):
                inputs = inputs.to(self.device)
                output = model(inputs)
                pred.append(output['pred'])
                gt.append(output['gt'])
            pred = torch.concat(pred).cpu().numpy()
            gt = torch.concat(gt).cpu().numpy()
        model.train()
        return compute_metrics(pred, gt)
    
    def main_one_iteration(self, args:CustomArgs, data:CustomData, training_iter_id=0):
        # === prepare === 
        if 1:
            # seed
            args.seed += training_iter_id
            set_seed(args.seed)
            # path
            train_fold_name = f'training_iteration_{training_iter_id}'
            args.ckpt_dir = os.path.join(args.ckpt_dir, train_fold_name)
            args.log_dir = os.path.join(args.log_dir, train_fold_name)
            args.check_path()
            
            logger = CustomLogger(
                log_dir=args.log_dir,
                logger_name=f'{args.cur_time}_iter{training_iter_id}_logger',
                print_output=True,
            )
            
            model = get_model(args.model, args.model_config)
            
            compute_metrics = ComputeMetrics(feature_list=data.feature_list)
        
        logger.log_json(dict(args), LOG_FILENAME_DICT['hyperparams'], log_info=False)

        # === train ===
        
        start_time = time.time()
        
        train_output = self.fit(
            args=args,
            model=model,
            data=data,
            compute_metrics=compute_metrics,
            logger=logger,
        )

        train_output['train_runtime'] = time.time()-start_time
        logger.log_json(train_output, LOG_FILENAME_DICT['output'], log_info=True, mode='w')
        
        if not args.save_ckpt:
            shutil.rmtree(args.ckpt_dir)

    def main(self, args:CustomArgs):
        from copy import deepcopy
        
        args.complete_path()
        args.check_path()
        
        data = CustomData(
            data_path=args.data_path,
            mini_dataset=args.mini_dataset,
        )
        args.trainset_size, args.devset_size, args.testset_size = map(len, [
            data.train_dataset, data.dev_dataset, data.test_dataset
        ])
        
        main_logger = CustomLogger(args.log_dir, logger_name=f'{args.cur_time}_main_logger', print_output=True)  
        main_logger.log_json(dict(args), LOG_FILENAME_DICT['hyperparams'], log_info=True)
        
        try:
            for training_iter_id in range(args.training_iteration):
                self.main_one_iteration(deepcopy(args), data=data, training_iter_id=training_iter_id)
            if not args.save_ckpt:
                shutil.rmtree(args.ckpt_dir)
        except Exception as _:
            error_file = main_logger.log_dir/'error.out'
            catch_and_record_error(error_file)
            exit(1)
        
        # calculate average
        for json_file_name in LOG_FILENAME_DICT.values():
            if json_file_name == LOG_FILENAME_DICT['hyperparams']:
                continue
            metric_analysis = analyze_metrics_json(args.log_dir, json_file_name, just_average=True)
            if metric_analysis:
                main_logger.log_json(metric_analysis, json_file_name, log_info=False)
        

if __name__ == '__main__':
    def local_test_args():
        args = CustomArgs(test_setting=True)
        
        args.version = 'test'
        args.server_name = 'local'
        
        args.data_path = path(__file__).parent.parent / r'data\data_96-96'
        args.data_path = str(args.data_path)
        
        args.model = 'lstm'
        args.ckpt_dir = './ckpt_space/'
        args.log_dir = './log_space/'

        args.model = 'transformer'
        args.model_config = TransformerConfig()
        return args
    
    args = local_test_args()
    args.prepare_gpu(target_mem_mb=1, gpu_cnt=1)
    args.save_ckpt = True
    Trainer().main(args)
    
    # args = CustomArgs()
    # main(args)