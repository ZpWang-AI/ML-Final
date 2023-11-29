import os
import json
import numpy as np
import pandas as pd
import datetime

from collections import defaultdict
from pathlib import Path as path


def analyze_metrics_json(log_dir, file_name, just_average=False):
    if path(file_name).suffix != '.json':
        return {}
    total_metrics = defaultdict(list)  # {metric_name: [values]}
    for dirpath, dirnames, filenames in os.walk(log_dir):
        if path(dirpath) == path(log_dir):
            continue
        for cur_file in filenames:
            if str(cur_file) == str(file_name):
                with open(path(dirpath, cur_file), 'r', encoding='utf8')as f:
                    cur_metrics = json.load(f)
                for k, v in cur_metrics.items():
                    total_metrics[k].append(v)
    if not total_metrics:
        return {}
    metric_analysis = {}
    for k, v in total_metrics.items():
        if just_average:
            metric_analysis[k] = np.mean(v)
        else:
            metric_analysis[k] = {
                'tot': v,
                'cnt': len(v),
                'mean': np.mean(v),
                'variance': np.var(v),
                'std': np.std(v),
                'error': np.std(v)/np.sqrt(len(v)),
                'min': np.min(v),
                'max': np.max(v),
                'range': np.max(v)-np.min(v),
            }
    return metric_analysis


if __name__ == '__main__':
    pass