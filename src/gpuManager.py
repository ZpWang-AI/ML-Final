import time
import pynvml

from typing import *


class GPUManager:
    @staticmethod
    def query_gpu_memory(cuda_id:Optional[int]=None, show=True, to_mb=True):
        if cuda_id is None:
            for p in GPUManager.get_all_cuda_id():
                GPUManager.query_gpu_memory(cuda_id=p, show=show, to_mb=to_mb)
            return
        
        def norm_mem(mem):
            if to_mb:
                return f'{mem/(1024**2):.0f}MB'
            unit_lst = ['B', 'KB', 'MB', 'GB', 'TB']
            for unit in unit_lst:
                if mem < 1024:
                    return f'{mem:.2f}{unit}'
                mem /= 1024
            return f'{mem:.2f}TB'
        
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        if show:
            print(
                f'cuda: {cuda_id}, '
                f'free: {norm_mem(info.free)}, '
                f'used: {norm_mem(info.used)}, '
                f'total: {norm_mem(info.total)}'
            )
        return info.free, info.used, info.total

    @staticmethod
    def get_all_cuda_id():
        pynvml.nvmlInit()
        cuda_cnt = list(range(pynvml.nvmlDeviceGetCount()))
        pynvml.nvmlShutdown()
        return cuda_cnt
        
    @staticmethod
    def _get_most_free_gpu(device_range=None):
        if not device_range:
            device_range = GPUManager.get_all_cuda_id()
        max_free = -1
        free_id = -1
        for cuda_id in device_range:
            cur_free = GPUManager.query_gpu_memory(cuda_id, show=False)[0]
            if cur_free > max_free:
                max_free = cur_free
                free_id = cuda_id
        return max_free, free_id
    
    @staticmethod
    def get_free_gpu(
        target_mem_mb=8000, 
        force=False, 
        wait=True, 
        wait_gap=5, 
        show_waiting=False,
        device_range=None, 
    ):
        if not device_range:
            device_range = GPUManager.get_all_cuda_id()

        if force:
            return GPUManager._get_most_free_gpu(device_range=device_range)[1]
            
        if not wait:
            target_mem_mb *= 1024**2
            # n = number of cuda device
            # query from cuda:n-1 to cuda:0
            for cuda_id in device_range[::-1]:
                if GPUManager.query_gpu_memory(cuda_id=cuda_id, show=False)[0] > target_mem_mb:
                    return cuda_id
            return -1
        else:
            while 1:
                device_id = GPUManager.get_free_gpu(
                    target_mem_mb=target_mem_mb,
                    force=False,
                    wait=False,
                    device_range=device_range,
                )
                if device_id != -1:
                    return device_id
                if show_waiting:
                    print('waiting cuda ...')
                time.sleep(wait_gap)
    
    @staticmethod
    def get_some_free_gpus(
        gpu_cnt=1,
        target_mem_mb=8000,
        device_range=None,
        return_str=True,
    ):
        if not device_range:
            device_range = GPUManager.get_all_cuda_id()

        target_mem_mb *= 1024**2
        while 1:
            gpu_id_lst = []
            # n = number of cuda device
            # query from cuda:n-1 to cuda:0            
            for cuda_id in device_range[::-1]:
                if GPUManager.query_gpu_memory(cuda_id=cuda_id, show=False)[0] > target_mem_mb:
                    gpu_id_lst.append(cuda_id)
                    if len(gpu_id_lst) >= gpu_cnt:
                        return ','.join(map(str,gpu_id_lst)) if return_str else gpu_id_lst
            time.sleep(5)
        

if __name__ == '__main__':
    GPUManager.query_gpu_memory(cuda_id=0, show=True)
    print(GPUManager.get_free_gpu(target_mem_mb=1))