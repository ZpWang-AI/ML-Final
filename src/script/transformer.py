# ===== prepare server_name, root_fold =====
SERVER_NAME = 'cu01_'
if SERVER_NAME in ['cu12_', 'cu13_', 'cu01_']:
    ROOT_FOLD = '/home/chongz/programFile/ML-Final-Test/ML-Final/'
# elif SERVER_NAME == :
#     ROOT_FOLD_IDRR = ''
else:
    raise Exception('wrong ROOT_FOLD_IDRR')

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
    args.ckpt_dir = '/home/chongz/programFile/ML-Final-Test/ML-Final/ckpt_space/'  # TODO: consume lots of memory
    if test_setting:
        args.log_dir = ROOT_FOLD+'log_space_test/'
    else:
        args.log_dir = ROOT_FOLD+'log_space/'

    return args


def server_experiment_args():
    args = server_base_args(test_setting=False)
    
    args.version = 'base'
    args.cuda_cnt = 1
    args.epochs = 10
    args.train_batch_size = 32
    args.eval_batch_size = 32
    args.eval_steps = 100
    args.log_steps = 5
    args.weight_decay = 0.01
    args.learning_rate = 1e-3
    args.save_ckpt = False
    
    # ===== TODO: modify args below, don't modify args above =====
    args.version = 'b'
    args.model_name = 'transformer'
    args.model_config = TransformerConfig(
        channels=128, num_layers=3, nhead=8, dropout=0.,
    )
    args.learning_rate = 1e-4
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
            show_model=True,
            show_server_name=False,
        )
            
        from main import Trainer
        Trainer().main(todo_args)
    
    def experiment_multi_times():
        cuda_cnt = 1  # === prepare gpu ===
        cuda_id = CustomArgs().prepare_gpu(target_mem_mb=10500, gpu_cnt=cuda_cnt) 
        from main import Trainer
        
        epochs = [10,15,20,25,30]
        millis = [1,5,10,15,20]
        for epoch, milli in zip(epochs, millis):
            todo_args = server_experiment_args()

            # === TODO: prepare args ===
            # todo_args.model_config = TransformerConfig(channels=channels,
            #                                            num_layers=num_layers,
            #                                            nhead=8,
            #                                            dropout=dropout)
            todo_args.version = f'epoch{epoch}_lr{milli}milli_batch_channels_numlayers_dropout_transformer'
            todo_args.epochs = epoch
            todo_args.learning_rate = float(f'{milli}e-4') 
            todo_args.train_batch_size = 32   
            # === TODO: prepare args ===
            
            todo_args.cuda_id = cuda_id
            todo_args.cuda_cnt = cuda_cnt
            todo_args.complete_path(
                show_cur_time=True,
                show_server_name=False,
            )
            
            Trainer().main(todo_args)

    # experiment_once()
    experiment_multi_times()
    pass