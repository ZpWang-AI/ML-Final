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
        

if __name__ == '__main__':
    pass