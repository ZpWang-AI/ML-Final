import os
import json
import transformers

from pathlib import Path as path
from typing import *
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from logger import CustomLogger


class CustomCallback(TrainerCallback):
    def __init__(
        self, 
        logger:CustomLogger,
        metric_names:list,
        evaluate_testdata=False,
    ):
        super().__init__()
        
        self.trainer:Trainer = None
        self.logger = logger
        self.evaluate_testdata = evaluate_testdata

        self.metric_names = metric_names
        self.best_metrics = dict(
            [('best_epoch_'+m,-1)for m in metric_names]+
            [('best_'+m,-1)for m in metric_names]
        )
        self.metric_map = {m:p for p, m in enumerate(self.metric_names)}
        
        self.best_metric_file_name = 'best_metric_score.json'
        self.dev_metric_file_name = 'dev_metric_score.jsonl'
        self.train_loss_file_name = 'train_loss.jsonl'
        
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logs = kwargs['logs']
        if 'loss' in logs and 'learning_rate' in logs:
            self.logger.log_json(kwargs['logs'], self.train_loss_file_name, log_info=False, mode='a')

    def on_evaluate(self, args, state, control, metrics:Dict[str, float], **kwargs):
        if self.evaluate_testdata:
            return
        
        dev_metrics = {'loss': metrics['eval_loss'], 'epoch': metrics['epoch']}
        for metric_name, metric_value in metrics.items():
            best_metric_name = metric_name.replace('eval_', 'best_')
            if best_metric_name not in self.best_metrics:
                continue
            dev_metrics[metric_name.replace('eval_', 'dev_')] = metric_value
            
            if metric_value > self.best_metrics[best_metric_name]:
                self.best_metrics[best_metric_name] = metric_value
                self.best_metrics[metric_name.replace('eval_', 'best_epoch_')] = metrics['epoch']
                
                best_model_path = path(args.output_dir)/f'checkpoint_{best_metric_name}'
                self.trainer.save_model(best_model_path)
                if self.logger:
                    self.logger.info(f'{best_metric_name}: {metric_value}')
                    # self.logger.info(f"New best model saved to {best_model_path}")

        self.logger.log_json(self.best_metrics, self.best_metric_file_name, log_info=False)
        self.logger.log_json(dev_metrics, self.dev_metric_file_name, log_info=True, mode='a')

            
    