import os 
import json
import shutil
import torch
import torch.nn as nn
import pandas as pd 
import numpy as np
import time

from typing import Any
from typing import *
from pathlib import Path as path
from transformers import set_seed

from utils import catch_and_record_error
from arguments import CustomArgs
from logger import CustomLogger
from data import CustomData, DataLoader
from model.LSTM import LSTM
from metrics import ComputeMetrics
from analyze import analyze_metrics_json

LOG_FILENAME_DICT = {
    'hyperparams': 'hyperparams.json',
    'best': 'best_metric_score.json',
    'dev': 'dev_metric_score.jsonl',
    'test': 'test_metric_score.json',
    'loss': 'train_loss.jsonl',
    'output': 'train_output.json',
}


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
        model.to(self.device)
        data.prepare_dataloader(train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size)
        train_batch = len(data.train_dataloader)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=train_batch*args.epochs)
        
        batch = 0
        model.train()
        total_loss, log_loss = 0, 0
        best_metrics = {m:-1 for m in compute_metrics.metric_names}
        
        for _ in range(args.epochs):
            for inputs in data.train_dataloader:
                batch += 1
                
                inputs = inputs.to(self.device)
                output = model(inputs)
                loss:torch.Tensor = output['loss']
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss
                log_loss += loss
                if not batch % args.log_steps:
                    cur_log = {
                        'loss': float((log_loss/args.log_steps).cpu()),
                        'lr': lr_scheduler.get_last_lr()[0],
                        'epoch': batch/train_batch,
                    }
                    log_loss = 0
                    logger.log_json(cur_log, LOG_FILENAME_DICT['loss'], log_info=True, mode='a')
                if not batch % args.eval_steps:
                    metrics = self.evaluate(model, data.dev_dataloader, compute_metrics)
                    if metrics['MSE'] > best_metrics['MSE']:
                        # TODO: save
                        pass
                    for k,v in metrics.items():
                        best_metrics[k] = max(best_metrics[k], v)
                    logger.log_json(metrics, LOG_FILENAME_DICT['dev'], log_info=True, mode='a')
                    logger.log_json(best_metrics, LOG_FILENAME_DICT['best'], log_info=True, mode='w')
        
        # TODO: model load
        test_metric = self.evaluate(model, data.test_dataloader, compute_metrics)
        logger.log_json(test_metric, LOG_FILENAME_DICT['test'], log_info=True, mode='w')
        
        return {'loss': float((total_loss/batch).cpu()), 'epoch':batch/train_batch}
    
    def evaluate(
        self,
        model:nn.Module,
        dataloader:DataLoader,
        compute_metrics:ComputeMetrics,
    ):
        model.eval()
        with torch.no_grad():
            pred, gt = [], []
            for inputs in dataloader:
                inputs = inputs.to(self.device)
                output = model.predict(inputs)
                pred.append(output['pred'])
                gt.append(output['gt'])
            pred = torch.concat(pred).cpu().numpy()
            gt = torch.concat(gt).cpu().numpy()
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
            
            model = LSTM(
                data_dim=7,
                hidden_size=128,
                num_layers=3,
                dropout=0.,
            )
            
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
                main_logger.log_json(metric_analysis, json_file_name, log_info=True)


if __name__ == '__main__':
    def local_test_args():
        args = CustomArgs(test_setting=True)
        
        args.version = 'test'
        args.server_name = 'local'
        
        args.data_path = r'D:\0--data\研究生学务\研一上\机器学习\Final\ML-Final\data\data_96-96'
        # args.data_path = r'D:\0--data\研究生学务\研一上\机器学习\Final\ML-Final\data\data_96-336'
        
        args.model = 'lstm'
        args.ckpt_dir = './ckpt_space/'
        args.log_dir = './log_space/'

        return args
    
    args = local_test_args()
    Trainer().main(args)
    
    # args = CustomArgs()
    # main(args)