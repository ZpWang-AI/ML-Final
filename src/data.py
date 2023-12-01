import json
import os
import numpy as np
import pandas as pd

from typing import *
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import Dataset
    

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
        self.columns = ['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        df = pd.read_csv(data_path)
        self.train_df: pd.DataFrame = None
        self.dev_df: pd.DataFrame = None
        self.test_df: pd.DataFrame = None
        self.get_dataframe(data_path=data_path, data_name=data_name)
        
        self.data_augmentation_train_df()
        if mini_dataset:
            self.train_df = self.train_df.iloc[:32]
            self.dev_df = self.dev_df.iloc[:16]
            self.test_df = self.test_df.iloc[:16]
            if data_name == 'conll':
                self.blind_test_df = self.blind_test_df.iloc[:16]

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        
        # label
        self.label_list: List[str] = []
        self.num_labels: int = 0
        self.label_map: Dict[str, int] = {}
        self.get_label_info(label_level=label_level, data_name=data_name)
        
        # dataset
        self.train_dataset = self.get_dataset(self.train_df, is_train=True)
        self.dev_dataset = self.get_dataset(self.dev_df, is_train=False)
        self.test_dataset = self.get_dataset(self.test_df, is_train=False)
        if data_name == 'conll':
            self.blind_test_dataset = self.get_dataset(self.blind_test_df)
        
        # data collator
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
    
    def get_dataframe(self, data_path, data_name):
        if data_name == 'pdtb2':
            df = pd.read_csv(
                data_path, 
                usecols=[
                    'Relation', 'Section', 
                    'Arg1_RawText', 'Arg2_RawText', 
                    'Conn1', 'Conn2',
                    'ConnHeadSemClass1', 'ConnHeadSemClass2',
                    'Conn2SemClass1', 'Conn2SemClass2'],
                low_memory=False,
            )
            df = df[df['Relation'] == 'Implicit']
            df = df.rename(columns={
                'Arg1_RawText': 'arg1',
                'Arg2_RawText': 'arg2',
                'Conn1': 'conn1',
                'Conn2': 'conn2',
                'ConnHeadSemClass1': 'conn1sense1',
                'ConnHeadSemClass2': 'conn1sense2',
                'Conn2SemClass1': 'conn2sense1',
                'Conn2SemClass2': 'conn2sense2',
            })
            
            self.train_df = df[~df['Section'].isin([0, 1, 21, 22, 23, 24])]
            self.dev_df = df[df['Section'].isin([0, 1])]
            self.test_df = df[df['Section'].isin([21, 22])]
            
        elif data_name == 'pdtb3':
            df = pd.read_csv(
                data_path,
                usecols=[
                    'section', 'relation_type',
                    'arg1', 'arg2', 
                    'conn1', 'conn2', 
                    'conn1_sense1', 'conn1_sense2',
                    'conn2_sense1', 'conn2_sense2',
                ],
                delimiter='\t',
                low_memory=False,
            )
            df = df[df['relation_type'] == 'Implicit']
            df = df.rename(columns={
                'conn1_sense1': 'conn1sense1',
                'conn1_sense2': 'conn1sense2',
                'conn2_sense1': 'conn2sense1',
                'conn2_sense2': 'conn2sense2',
            })
            
            self.train_df = df[~df['section'].isin([0, 1, 21, 22, 23, 24])]
            self.dev_df = df[df['section'].isin([0, 1])]
            self.test_df = df[df['section'].isin([21, 22])]
            
        elif data_name == 'conll':
            def load_conll_json(json_path):
                with open(json_path, 'r', encoding='utf8')as f:
                    init_dicts = [json.loads(line)for line in f.readlines()]
                dicts = []
                for dic in init_dicts:
                    if dic['Type'] == 'Implicit':
                        cur_dic = {
                            'arg1':dic['Arg1']['RawText'],
                            'arg2':dic['Arg2']['RawText'],
                            'conn1':dic['Connective']['RawText'],
                            'conn1sense1':dic['Sense'][0],
                        }
                        if len(dic['Sense']) > 1:
                            cur_dic['conn1sense2'] = dic['Sense'][1]
                        dicts.append(cur_dic)
                return pd.DataFrame(dicts, columns=self.columns)
            
            self.train_df = load_conll_json(os.path.join(data_path, r'train.json'))
            self.dev_df = load_conll_json(os.path.join(data_path, r'dev.json'))
            self.test_df = load_conll_json(os.path.join(data_path, r'test.json'))
            self.blind_test_df = load_conll_json(os.path.join(data_path, r'blind-test.json'))

        else:
            raise Exception('wrong data_name')
        
    def get_label_info(self, label_level, data_name):
        if label_level == 'level1':
            self.label_list = [
                "Comparison",
                "Contingency",
                "Expansion",
                "Temporal"
            ]
        elif label_level == 'level2':
            if data_name == 'pdtb2':
                self.label_list = [
                    "Comparison.Concession",
                    "Comparison.Contrast",
                    "Contingency.Cause",
                    "Contingency.Pragmatic cause",
                    "Expansion.Alternative",
                    "Expansion.Conjunction",
                    "Expansion.Instantiation",
                    "Expansion.List",
                    "Expansion.Restatement",
                    "Temporal.Asynchronous",
                    "Temporal.Synchrony"
                ]
            elif data_name == 'pdtb3':
                self.label_list = [
                    "Comparison.Concession",
                    "Comparison.Contrast",
                    "Contingency.Cause",
                    "Contingency.Pragmatic cause",
                    "Expansion.Alternative",
                    "Expansion.Conjunction",
                    "Expansion.Instantiation",
                    "Expansion.List",
                    "Expansion.Restatement",
                    "Temporal.Asynchronous",
                    "Temporal.Synchrony"
                ]
            # elif data_name == 'conll':
            #     self.label_list = [
            #         'Comparison.Concession',
            #         'Comparison.Contrast',
            #         'Contingency.Cause',
            #         'Contingency.Condition',
            #         'Expansion.Alternative',
            #         'Expansion.Conjunction',
            #         'Expansion.Exception',
            #         'Expansion.Instantiation',
            #         'Expansion.Restatement',
            #         'Temporal.Asynchronous',
            #         'Temporal.Synchrony'
            #     ]
        
        if not self.label_list:
            raise ValueError('wrong label_level or wrong data_name')
        self.num_labels = len(self.label_list)
        self.label_map = {label:p for p, label in enumerate(self.label_list)}
        
    def label_to_id(self, sense):
        """
        return -1 if sense is invalid or useless
        """
        if pd.isna(sense):
            return -1
        if self.label_level == 'level1':
            sense_l1 = sense.split('.')[0]
            label_id = self.label_map.get(sense_l1, -1)
        elif self.label_level == 'level2':
            sense_l2 = '.'.join(sense.split('.')[:2])
            label_id = self.label_map.get(sense_l2, -1)
        else:
            raise ValueError('wrong label_level')
        return label_id
    
    def get_dataset(self, df, is_train=False):
        arg1_list, arg2_list = [], []
        label_ids = []
        secondary_label_ids = []
        for p in range(df.shape[0]):
            row = df.iloc[p]
            arg1 = row.arg1
            arg2 = row.arg2
            conn1 = row.conn1
            conn2 = row.conn2
            conn1sense1 = row.conn1sense1
            conn1sense2 = row.conn1sense2
            conn2sense1 = row.conn2sense1
            conn2sense2 = row.conn2sense2
            
            primary_label = self.label_to_id(conn1sense1)
            if primary_label == -1:
                continue
            cur_sec_labels = [self.label_to_id(sense)
                              for sense in [conn1sense2, conn2sense1, conn2sense2]]
            cur_sec_labels = list(filter(lambda x:x!=-1, cur_sec_labels))
            
            arg1_list.append(arg1)
            arg2_list.append(arg2)
            label_ids.append(primary_label)
            secondary_label_ids.append(cur_sec_labels)
        
        label_vectors = np.eye(self.num_labels)[label_ids]
        if is_train:
            if self.secondary_label_weight != 0:
                for p, sec_labels in enumerate(secondary_label_ids):
                    for sl in sec_labels:
                        label_vectors[p][sl] = self.secondary_label_weight
        else:
            for p, sec_labels in enumerate(secondary_label_ids):
                for sl in sec_labels:
                    label_vectors[p][sl] = 1           
            
        return CustomDataset(
            arg1=arg1_list,
            arg2=arg2_list,
            tokenizer=self.tokenizer,
            # labels=label_ids,
            labels=label_vectors,
        )
                        
    def data_augmentation_train_df(self):
        if self.data_augmentation_secondary_label:
            df = self.train_df
            df2 = df.copy()
            df2.dropna(subset=['conn1sense2'], inplace=True)
            df2['conn1sense1'] = df2['conn1sense2']
            df3 = df.copy()
            df3.dropna(subset=['conn2sense1'], inplace=True)
            df3['conn1sense1'], df3['conn1'] = df3['conn2sense1'], df3['conn2']
            df4 = df.copy()
            df4.dropna(subset=['conn2sense2'], inplace=True)
            df4['conn1sense1'], df4['conn1'] = df4['conn2sense2'], df4['conn2']
            self.train_df = pd.concat([df,df2,df3,df4], ignore_index=True)
            for dv in 'conn2 conn1sense2 conn2sense1 conn2sense2'.split():
                self.train_df[dv] = np.nan
            
        if self.data_augmentation_connective_arg2:
            df = self.train_df
            df2 = df.copy()
            df2['arg1'] = df2['conn1']+df2['arg2']
            df3 = df.copy()
            df3.dropna(subset=['conn2'], inplace=True)
            df3['arg2'] = df3['conn2']+df3['arg2']
            df3['conn1sense1'], df3['conn1sense2'], df3['conn2sense1'], df3['conn2sense2'] = (
                df3['conn2sense1'], df3['conn2sense2'], df3['conn1sense1'], df3['conn1sense2']
            )
            self.train_df = pd.concat([df, df2, df3], ignore_index=True)
    
    
if __name__ == '__main__':
    import os, time
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def sampling_test(data_name, label_level='level1'):
        start_time = time.time()
        if data_name == 'pdtb2':
            data_path = r'D:\0--data\projects\04.01-IDRR\IDRR-base\CorpusData\PDTB2\pdtb2.csv'
        elif data_name == 'pdtb3':
            data_path = r'D:\0--data\projects\04.01-IDRR\IDRR-base\CorpusData\PDTB3\pdtb3_implicit.csv'
        elif data_name == 'conll':
            data_path = r'D:\0--data\projects\04.01-IDRR\IDRR-base\CorpusData\CoNLL16'
        else:
            raise Exception('wrong data_name')
            
        sample_dataset = CustomCorpusData(
            data_path=data_path,
            data_name=data_name,
            model_name_or_path=r'D:\0--data\projects\04.01-IDRR\IDRR-base\plm_cache\models--roberta-base\snapshots\bc2764f8af2e92b6eb5679868df33e224075ca68',
            # cache_dir='./plm_cache/',
            label_level=label_level,
            mini_dataset=False,
            data_augmentation_secondary_label=True,
            data_augmentation_connective_arg2=False,
        )
        batch = [sample_dataset.train_dataset[p]for p in range(3)]
        batch = sample_dataset.data_collator(batch)
        print(batch ,'\n')
        print(f'{data_name}, time: {time.time()-start_time:.2f}s')
        print(sample_dataset.train_df.shape)
        print('='*10)
    
    # sampling_test('pdtb2')
    # sampling_test('pdtb2', 'level2')
    # sampling_test('pdtb3')
    # sampling_test('pdtb3', 'level2')
    # sampling_test('conll')
    
    # from run import local_test_args
    # sample_data = CustomCorpusData(**dict(local_test_args()))
    # print(len(sample_data.train_dataset))
    pass