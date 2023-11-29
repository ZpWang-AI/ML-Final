import os 
import json
import shutil
import torch
import pandas as pd 
import numpy as np
import time

from typing import *
from pathlib import Path as path
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, set_seed

from utils import catch_and_record_error
from arguments import CustomArgs
from logger import CustomLogger
from data import CustomData
from model.transformer import CustomModel
from metrics import ComputeMetrics
from callbacks import CustomCallback
from analyze import analyze_metrics_json

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

LOG_FILENAME_DICT = {
    'hyperparams': 'hyperparams.json',
    'best': 'best_metric_score.json',
    'dev': 'dev_metric_score.jsonl',
    'test': 'test_metric_score.json',
    'blind test': 'test_blind_metric_score.json',
    'loss': 'train_loss.jsonl',
    'output': 'train_output.json',
}


def train_func(
    args:CustomArgs, 
    training_args:TrainingArguments, 
    logger:CustomLogger,
    data:CustomData, 
    model:CustomModel, 
    compute_metrics:ComputeMetrics,
):
    callback = CustomCallback(
        logger=logger, 
        metric_names=compute_metrics.metric_names,
    )
    callback.best_metric_file_name = LOG_FILENAME_DICT['best']
    callback.dev_metric_file_name = LOG_FILENAME_DICT['dev']
    callback.train_loss_file_name = LOG_FILENAME_DICT['loss']
    
    trainer = Trainer(
        model=model, 
        args=training_args, 
        tokenizer=data.tokenizer, 
        compute_metrics=compute_metrics,
        callbacks=[callback],
        data_collator=data.data_collator,
        
        train_dataset=data.train_dataset,
        eval_dataset=data.dev_dataset, 
    )
    callback.trainer = trainer

    train_output = trainer.train().metrics
    logger.log_json(train_output, LOG_FILENAME_DICT['output'], log_info=True)
    final_state_fold = path(training_args.output_dir)/'final'
    trainer.save_model(final_state_fold)
    
    # do test 
    callback.evaluate_testdata = True
    
    test_metrics = {}
    for metric_ in compute_metrics.metric_names:
        load_ckpt_dir = path(training_args.output_dir)/f'checkpoint_best_{metric_}'
        if load_ckpt_dir.exists():
            model.load_state_dict(torch.load(load_ckpt_dir/'pytorch_model.bin'))
            evaluate_output = trainer.evaluate(eval_dataset=data.test_dataset)
            test_metrics['test_'+metric_] = evaluate_output['eval_'+metric_]
            
    logger.log_json(test_metrics, LOG_FILENAME_DICT['test'], log_info=True)                
    # model.load_state_dict(torch.load(final_state_fold/'pytorch_model.bin'))
    
    # if args.data_name == 'conll':
    #     test_metrics = {}
    #     for metric_ in compute_metrics.metric_names:
    #         load_ckpt_dir = path(args.output_dir)/f'checkpoint_best_{metric_}'
    #         if load_ckpt_dir.exists():
    #             evaluate_output = trainer.evaluate(eval_dataset=data.blind_test_dataset)
    #             test_metrics['test_'+metric_] = evaluate_output['eval_'+metric_]
                
    #     logger.log_json(test_metrics, LOG_FILENAME_DICT['blind test'], log_info=True)    

    return trainer, callback


def main_one_iteration(args:CustomArgs, data:CustomData, training_iter_id=0):
    # === prepare === 
    if 1:
        # seed
        args.seed += training_iter_id
        set_seed(args.seed)
        # path
        train_fold_name = f'training_iteration_{training_iter_id}'
        args.output_dir = os.path.join(args.output_dir, train_fold_name)
        args.log_dir = os.path.join(args.log_dir, train_fold_name)
        args.check_path()
        
        training_args = TrainingArguments(
            output_dir = args.output_dir,
            
            # strategies of evaluation, logging, save
            evaluation_strategy = "steps", 
            eval_steps = args.eval_steps,
            logging_strategy = 'steps',
            logging_steps = args.log_steps,
            save_strategy = 'no',
            # save_strategy = 'epoch',
            # save_total_limit = 1,
            
            # optimizer and lr_scheduler
            optim = 'adamw_torch',
            learning_rate = args.learning_rate,
            weight_decay = args.weight_decay,
            lr_scheduler_type = 'linear',
            warmup_ratio = args.warmup_ratio,
            
            # epochs and batches 
            num_train_epochs = args.epochs, 
            max_steps = args.max_steps,
            per_device_train_batch_size = args.train_batch_size,
            per_device_eval_batch_size = args.eval_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
        )
        
        logger = CustomLogger(
            log_dir=args.log_dir,
            logger_name=f'{args.cur_time}_iter{training_iter_id}_logger',
            print_output=True,
        )
        
        model = CustomModel(
            model_name_or_path=args.model_name_or_path,
            num_labels=data.num_labels,
            cache_dir=args.cache_dir,
            loss_type=args.loss_type,
        )
        
        compute_metrics = ComputeMetrics(label_list=data.label_list)
        
        train_evaluate_kwargs = {
            'args': args,
            'training_args': training_args,
            'model': model,
            'data': data,
            'compute_metrics': compute_metrics,
            'logger': logger,
        }
    
    logger.log_json(dict(args), LOG_FILENAME_DICT['hyperparams'], log_info=False)

    # === train ===
    
    start_time = time.time()
    
    train_func(**train_evaluate_kwargs)

    total_runtime = time.time()-start_time
    with open(logger.log_dir/LOG_FILENAME_DICT['output'], 'r', encoding='utf8')as f:
        train_output = json.load(f)
        train_output['train_runtime'] = total_runtime
    logger.log_json(train_output, LOG_FILENAME_DICT['output'], False)
    
    if not args.save_ckpt:
        shutil.rmtree(args.output_dir)


def main(args:CustomArgs, training_iter_id=-1):
    """
    params:
        args: CustomArgs
        training_iter_id: int ( set t=args.training_iteration )
            -1: auto train t iterations
            0, 1, ..., t-1: train a specific iteration
            t: calculate average of metrics
    """
    from copy import deepcopy
    
    args.complete_path()
    args.check_path()
    
    data = CustomData(
        data_path=args.data_path,
        data_name=args.data_name,
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
        label_level=args.label_level,
        secondary_label_weight=args.secondary_label_weight,
        mini_dataset=args.mini_dataset,
        data_augmentation_secondary_label=args.data_augmentation_secondary_label,
        data_augmentation_connective_arg2=args.data_augmentation_connective_arg2,
    )
    args.trainset_size, args.devset_size, args.testset_size = map(len, [
        data.train_dataset, data.dev_dataset, data.test_dataset
    ])
    args.recalculate_eval_log_steps()
    
    main_logger = CustomLogger(args.log_dir, logger_name=f'{args.cur_time}_main_logger', print_output=True)
    if training_iter_id < 0 or training_iter_id == 0:    
        main_logger.log_json(dict(args), log_file_name=LOG_FILENAME_DICT['hyperparams'], log_info=True)
    
    try:
        if training_iter_id < 0:
            for _training_iter_id in range(args.training_iteration):
                main_one_iteration(deepcopy(args), data=data, training_iter_id=_training_iter_id)
            if not args.save_ckpt:
                shutil.rmtree(args.output_dir)
        else:
            main_one_iteration(deepcopy(args), data=data, training_iter_id=training_iter_id)
    except Exception as e:
        error_file = main_logger.log_dir/'error.out'
        catch_and_record_error(error_file)
        exit(1)
    
    if training_iter_id < 0 or training_iter_id == args.training_iteration:
        # calculate average
        for json_file_name in LOG_FILENAME_DICT.values():
            if json_file_name == LOG_FILENAME_DICT['hyperparams']:
                continue
            metric_analysis = analyze_metrics_json(args.log_dir, json_file_name, just_average=True)
            if metric_analysis:
                main_logger.log_json(metric_analysis, json_file_name, log_info=True)


if __name__ == '__main__':
    def local_test_args(data_name='pdtb2', label_level='level1'):
        args = CustomArgs(test_setting=True)
        
        args.version = 'test'
        args.server_name = 'local'
        
        args.data_name = data_name
        if data_name == 'pdtb2':
            args.data_path = './CorpusData/PDTB2/pdtb2.csv'
        elif data_name == 'pdtb3':
            args.data_path = './CorpusData/PDTB3/pdtb3_implicit.csv'
        elif data_name == 'conll':
            args.data_path = './CorpusData/CoNLL16/'  
        
        args.model_name_or_path = './plm_cache/models--roberta-base/snapshots/bc2764f8af2e92b6eb5679868df33e224075ca68/'
        args.cache_dir = './plm_cache/'
        args.output_dir = './output_space/'
        args.log_dir = './log_space/'

        return args
    
    args = local_test_args()
    main(args)
    
    # args = CustomArgs()
    # main(args)