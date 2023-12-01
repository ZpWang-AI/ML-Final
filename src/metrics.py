import numpy as np


def cal_MSE(mat1, mat2):
    return ((mat1-mat2)**2).sum() / len(mat1)

def cal_MAE(mat1, mat2):
    return (np.abs(mat1-mat2)).sum() / len(mat1)


class ComputeMetrics:
    def __init__(self, feature_list:list) -> None:
        self.feature_list = feature_list
        self.metric_names = ['MSE', 'MAE']  # +feature_list
        
    def __call__(self, pred, gt):
        """
        n = data dimension
        eval_pred: (predictions, labels)
        predictions: np.array [datasize, n]
        labels: np.array [datasize, n]
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
    