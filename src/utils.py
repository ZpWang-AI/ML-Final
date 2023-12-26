import os
import shutil
import time 
import logging
import json
import numpy as np
import traceback

from pathlib import Path as path
from typing import *


def catch_and_record_error(error_file):
    with open(error_file, 'w', encoding='utf8')as f:
        error_string = traceback.format_exc()
        f.write(error_string)
        print(f"\n{'='*10} ERROR {'='*10}\n")
        print(error_string)
        print(f"\n{'='*10} ERROR {'='*10}\n")
        
        
def count_parameters(model):
    param_cnt = sum(p.numel() for p in model.parameters())
    param_cnt = str(param_cnt)
    cnt_n = len(param_cnt)
    param_list = [param_cnt[max(0, p-3):p]for p in range(cnt_n, 0, -3)]
    param_str = ','.join(param_list[::-1])
    return param_str


if __name__ == '__main__':
    pass