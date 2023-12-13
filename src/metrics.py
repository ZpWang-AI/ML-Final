import numpy as np


def cal_MSE(mat1, mat2):
    mse = ((mat1-mat2)**2).mean(axis=(0,1)).sum()
    return float(mse)

def cal_MAE(mat1, mat2):
    mae = (np.abs(mat1-mat2)).mean(axis=(0,1)).sum()
    return float(mae)


class ComputeMetrics:
    def __init__(self, feature_list:list) -> None:
        self.feature_list = feature_list
        self.metric_names = ['MSE', 'MAE']  # +feature_list
        
    def __call__(self, pred, gt):
        """
        s = sequence length
        n = data dimension (7 or 8)
        pred: np.array [datasize, s, n]
        gt: np.array [datasize, s, n]
        """
        
        res = {
            # 'Acc': accuracy_score(labels, predictions),
            'MSE': cal_MSE(pred, gt),
            'MAE': cal_MAE(pred, gt),
        }
        
        # for i, target_type in enumerate(self.feature_list):
        #     res[target_type] = cal_MSE(predictions[:])
        
        return res
    
    
if __name__ == '__main__':
    pass
    