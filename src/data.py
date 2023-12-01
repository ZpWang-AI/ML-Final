import json
import os
import torch
import numpy as np
import pandas as pd

from typing import Any
from typing import *
from pathlib import Path as path
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import Dataset
    
    
class CustomDataset(Dataset):
    def __init__(self, df) -> None:
        self.df:pd.DataFrame = df
    
    def __getitem__(self, index) -> Any:
        return self.df.iloc[index]
    
    def __len__(self):
        return len(self.df)


class CustomDataCollator:
    def __call__(self, features):
        """
        features: List[List[]]
        """
        features = torch.tensor(np.array(features))
        features = features.reshape(features.shape[0], -1, 8)
        return features


class CustomData:
    def __init__(
        self, 
        data_path,
        mini_dataset=False,
        *args, **kwargs,
    ):
        # args
        self.mini_dataset = mini_dataset
        
        # dataframe
        self.train_df = pd.read_csv(path(data_path, 'train.csv'), index_col=0)
        self.dev_df = pd.read_csv(path(data_path, 'dev.csv'), index_col=0)
        self.test_df = pd.read_csv(path(data_path, 'test.csv'), index_col=0)

        if mini_dataset:
            self.train_df = self.train_df.iloc[:32]
            self.dev_df = self.dev_df.iloc[:16]
            self.test_df = self.test_df.iloc[:16]

        # feature
        self.feature_list = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        self.feature_num = len(self.feature_list)
        self.feature_map = {f:p for p,f in enumerate(self.feature_list)}
        
        # dataset
        self.train_dataset = CustomDataset(self.train_df)
        self.dev_dataset = CustomDataset(self.dev_df)
        self.test_dataset = CustomDataset(self.test_df)
        
        # data collator
        self.data_collator = CustomDataCollator()
              
              
if __name__ == '__main__':
    import os, time

    start_time = time.time()
    
    sample_dataset = CustomData(
        r'D:\0--data\研究生学务\研一上\机器学习\Final\tmp\data_96-96',
        mini_dataset=False,
    )
    batch = [sample_dataset.train_dataset[p]for p in range(3)]
    batch = sample_dataset.data_collator(batch)
    print('batch shape', batch.shape, '\n')
    print(f'time: {time.time()-start_time:.2f}s')
    print('train size', len(sample_dataset.train_dataset))
    pass