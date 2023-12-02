import os
import json

from pathlib import Path as path
from datetime import datetime


def fill_with_delimiter(s):
    return f'{"="*10} {s} {"="*(30-len(s))}'


class CustomArgs:
    def __init__(self, test_setting=False) -> None:
        self.version = 'base'
        
        # ========== base setting ==================
        self.part1 = 'base setting'
        self.save_ckpt = False
        self.seed = 2023
        
        self.training_iteration = 5
        self.cuda_cnt = 1
        
        # ========== data ==========================
        self.part2 = 'data'
        self.mini_dataset = False

        self.trainset_size = -1
        self.devset_size = -1
        self.testset_size = -1
        
        # ========== model =========================
        self.part3 = 'model'
        self.model = 'lstm'
        self.hidden_size = 128
        self.num_layers = 2
        
        # ========== loss ==========================
        self.part4 = 'loss'
        self.loss_type = 'MSELoss'
        
        # ========== epoch, batch, step ============
        self.part5 = 'epoch, batch, step'
        self.epochs = 5
        self.train_batch_size = 8
        self.eval_batch_size = 32
        self.eval_steps = 100
        self.log_steps = 10
        
        # ========== lr ============================
        self.part6 = 'lr'
        self.weight_decay = 0.01
        self.learning_rate = 5e-6
        
        # ========== file path =====================
        self.part7 = 'file path'
        self.data_path = './data/data_96-96'
        self.cache_dir = 'plm_cache'
        self.ckpt_dir = 'ckpt_space'
        self.log_dir = 'log_space'
        
        # ========== additional details ============
        self.part8 = 'additional details'
        self.cuda_id = ''
        self.cur_time = ''
        self.server_name = ''
        
        for p in range(1, 100):
            attr_name = f'part{p}'
            if hasattr(self, attr_name):
                init_attr = self.__getattribute__(attr_name)
                self.__setattr__(attr_name, fill_with_delimiter(init_attr))
        
        if test_setting:
            self.version = 'test'
            self.mini_dataset = True
            self.training_iteration = 2
            self.train_batch_size = 8
            self.eval_batch_size = 8
            self.epochs = 2
            self.eval_steps = 4
            self.log_steps = 4
            self.eval_per_epoch = -1
    
    def complete_path(self,
                      show_cur_time=True,
                      show_server_name = True,
                      show_data_name=True,
                      show_label_level=True,
                      ):
        if not self.cur_time:
            self.cur_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            specific_fold_name = []
            if show_cur_time:
                specific_fold_name.append(self.cur_time)
            if show_server_name:
                specific_fold_name.append(self.server_name)
            specific_fold_name.append(self.version)
            if show_data_name:
                specific_fold_name.append(self.data_name)
            if show_label_level:
                specific_fold_name.append(self.label_level)
            specific_fold_name = '.'.join(map(str, specific_fold_name))
            self.ckpt_dir = os.path.join(self.ckpt_dir, specific_fold_name)
            self.log_dir = os.path.join(self.log_dir, specific_fold_name) 
    
    def estimate_cuda_memory(self):
        return 10500
        if self.train_batch_size > 16:
            return 10500
        elif self.train_batch_size > 8:
            return 7000
        else:
            return 5000
    
    def prepare_gpu(self, target_mem_mb=10000, gpu_cnt=None):
        if not self.cuda_id:
            if target_mem_mb < 0:
                target_mem_mb = self.estimate_cuda_memory()
            if gpu_cnt is None:
                gpu_cnt = self.cuda_cnt

            from gpuManager import GPUManager
            free_gpu_ids = GPUManager.get_some_free_gpus(
                gpu_cnt=gpu_cnt, 
                target_mem_mb=target_mem_mb,
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = free_gpu_ids
            self.cuda_id = free_gpu_ids
            print(f'=== CUDA {free_gpu_ids} ===')
        return self.cuda_id
            
    def check_path(self):
        # self.data_path
        # self.cache_dir
        # self.ckpt_dir
        # self.log_dir
        # self.load_ckpt_dir
        
        assert path(self.data_path).exists(), 'wrong data path'
        
        if str(self.cache_dir) in ['None', '']:
            self.cache_dir = None
        else:
            path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            
        path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)
        path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # if not self.do_train and self.do_eval and not path(self.load_ckpt_dir).exists():
        #     raise Exception('no do_train and load_ckpt_dir does not exist')  
        
    def __iter__(self):
        return iter(self.__dict__.items())
    
    def __str__(self) -> str:
        return json.dumps(dict(self), ensure_ascii=False, indent=2)
        

if __name__ == '__main__':
    sample_args = CustomArgs(test_setting=False)
    print(sample_args)