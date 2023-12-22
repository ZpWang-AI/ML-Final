# ===== TODO: prepare server_name, root_fold, ckpt_space =====
SERVER_NAME = 'cu01_'  # doesn't matter, just for record. end with "_"
ROOT_FOLD = '/home/chongz/programFile/ML-Final-Test/ML-Final/'  # where this repository is located
CKPT_SPACE = '/home/chongz/programFile/ML-Final-Test/ML-Final/ckpt_space/'  # where checkpoints are saved. consume lots of memory
# ============================================================

import os, sys
CODE_SPACE = ROOT_FOLD+'src/'
if __name__ == '__main__':
    os.chdir(CODE_SPACE)
    sys.path.insert(0, CODE_SPACE)

from arguments import CustomArgs
from model.configs import *


def server_base_args(test_setting=False) -> CustomArgs:
    args = CustomArgs(test_setting=test_setting)
    
    args.version = 'test' if test_setting else 'base'
    args.server_name = SERVER_NAME
    
    # file path
    args.data_path = ROOT_FOLD+f'data/data_96-96/'
    args.cache_dir = ''
    args.ckpt_dir = CKPT_SPACE
    if test_setting:
        args.log_dir = ROOT_FOLD+'log_space_test/'
    else:
        args.log_dir = ROOT_FOLD+'log_space/'

    return args


def server_experiment_args():
    args = server_base_args(test_setting=False)
    
    args.version = 'base-lstm-multi'
    args.cuda_cnt = 1
    args.epochs = 10
    args.train_batch_size = 32
    args.eval_batch_size = 32
    args.eval_steps = 100
    args.log_steps = 5
    args.weight_decay = 0.01
    args.learning_rate = 1e-3
    
    # ===== TODO: modify args below, don't modify args above =====
    args.model = 'lstm'
    # args.model_config = TransformerConfig(
    #     channels=128, num_layers=3, nhead=8, dropout=0.,
    # )
    # ============================================================
    return args
    
    
if __name__ == '__main__':
    def experiment_once():
        todo_args = server_base_args(test_setting=True)
        todo_args = server_experiment_args()
        
        # todo_args.prepare_gpu(target_mem_mb=10000)  # when gpu usage is low
        todo_args.prepare_gpu(target_mem_mb=-1)
        todo_args.complete_path(
            show_cur_time=True,
            show_server_name=False,
        )
            
        from main import Trainer
        Trainer().main(todo_args)
    
    def experiment_multi_times():
        cuda_cnt = 1  # === prepare gpu ===
        cuda_id = CustomArgs().prepare_gpu(target_mem_mb=10500, gpu_cnt=cuda_cnt) 
        from main import Trainer
        
        # ===== TODO: set loop =====
        for epoch in range(10,31):
            for milli in [1,5,10,15,20]:
                for batch_size in [24,32,36,48,64,96]:
        # ==========================
        
                    todo_args = server_experiment_args()

                    # ===== TODO: prepare args =====
                    todo_args.version = f'epoch{epoch}_lr{milli}milli_batch{batch_size}_lstm_cu01'
                    todo_args.epochs = epoch
                    todo_args.learning_rate = float(f'{milli}e-3') 
                    todo_args.train_batch_size = batch_size  
                    # ==============================
                    
                    todo_args.cuda_id = cuda_id
                    todo_args.cuda_cnt = cuda_cnt
                    todo_args.complete_path(
                        show_cur_time=True,
                        show_server_name=False,
                    )
                    
                    Trainer().main(todo_args)

    # ===== TODO: choose func to do experiments =====
    # experiment_once()
    experiment_multi_times()
    # ===============================================
    pass