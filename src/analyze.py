import os
import json
import numpy as np
import pandas as pd
import datetime

from collections import defaultdict
from pathlib import Path as path


def analyze_metrics_json(log_dir, file_name):
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
    metric_analysis = dict(
        [(k, np.mean(v))for k,v in total_metrics.items()] +
        [(k+'_std', np.std(v))for k,v in total_metrics.items()]
    )
    return metric_analysis


if __name__ == '__main__':
    pass