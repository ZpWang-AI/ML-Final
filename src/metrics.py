import numpy as np

from sklearn.metrics import f1_score, accuracy_score


class ComputeMetrics:
    def __init__(self, label_list:list) -> None:
        self.label_list = label_list
        self.metric_names = ['Macro-F1', 'Acc']+label_list
        
    def __call__(self, eval_pred):
        """
        n = label categories
        eval_pred: (predictions, labels)
        predictions: np.array [datasize, n]
        labels: np.array [datasize, n]
        X[p][q]=True, sample p belongs to label q (False otherwise)
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        predictions = np.eye(len(self.label_list))[predictions]
        labels = (labels != 0).astype(int)
        
        res = {
            # 'Acc': accuracy_score(labels, predictions),
            'Macro-F1': f1_score(labels, predictions, average='macro', zero_division=0),
            'Acc': np.sum(predictions*labels)/len(predictions),
        }
        
        for i, target_type in enumerate(self.label_list):
            res[target_type] = f1_score(predictions[:,i], labels[:,i], zero_division=0)
        
        return res
    
    
if __name__ == '__main__':
    pass
    