import os,sys
import json
from pathlib import Path as path

root_path = '/home/chongz/programFile/ML-Final-Test/ML-Final/'

# define the code running path
code_run_path = root_path

if not os.path.exists(path(code_run_path + 'experiment/experiment_analysis/result/')):
    os.makedirs(path(code_run_path + 'experiment/experiment_analysis/result/'))

assert os.path.exists(path(code_run_path + 'experiment/experiment_analysis/result/')), "path not exists"

result_root_path = code_run_path + 'experiment/experiment_analysis/result/'

def get_best_experiments(experiments_res_path, mode:str):
    """
    experiment_res_path: the log path
    mode: model name
    """
    # dump result
    best_metric = {"best_MSE": 10000000000.0,"best_MAE": 10000000000.0}
    test_metric = {"test_MSE": 10000000000.0,"test_MAE": 10000000000.0}
    hyperparams = {}
    train_output = {}
    res_test_path, res_best_path = "", ""
    # tmp
    best_metric_tmp = {}
    test_metric_tmp = {}
    hyperparams_tmp = {}
    train_output_tmp = {}

    # agg result
    outer_folders = os.listdir(experiments_res_path)
    
    for outer_folder in outer_folders: # get all kinds of hyperparameters--folder

        # select the corresponding model result, if model_name and model retule not correspond,then next
        if outer_folder.split("_")[-1] != mode:
            continue
        outer_path = experiments_res_path + outer_folder + '/'
        
        inner_folders = os.listdir(outer_path)
        # inner_folders = inner_folders[5:]

        for inner_res_file in inner_folders: # get all result file
            inner_res_file_path = outer_path + inner_res_file
            if inner_res_file.endswith(".json"):
                with open(inner_res_file_path) as file:
                    result_json = json.load(file)
                    if inner_res_file.startswith("best"):
                        best_metric_tmp = result_json
                    elif inner_res_file.startswith("test"):
                        test_metric_tmp = result_json
                    elif inner_res_file.startswith("hyper"):
                        hyperparams_tmp = result_json
                    elif inner_res_file.startswith("train"):
                        train_output_tmp = result_json
        t_mse,t_mae,t_mse_t,t_mae_t = (test_metric['test_MSE'],test_metric["test_MAE"],
                                      test_metric_tmp['test_MSE'],test_metric_tmp['test_MAE'] )
        # 至少有一个指标更小且至多不能有指标更大
        if (t_mse_t < t_mse and t_mae_t <= t_mae) or (t_mse_t <= t_mse and t_mae_t < t_mae):
            test_metric,best_metric,hyperparams,train_output = test_metric_tmp,best_metric_tmp, hyperparams_tmp,train_output_tmp
    agg_res = {
        "test_metric": test_metric,
        "best_metric": best_metric,
        "train_output": train_output,
        "hyperparams": hyperparams
    }
    with open(path(code_run_path + 'experiment/experiment_analysis/result/' + mode + '_'+ outer_folder + '.json'), 'w') as file:
        json.dump(agg_res, file, indent=4)
            



if __name__ == '__main__':
    os.chdir(code_run_path)
    sys.path.insert(0, code_run_path)
    experiments_res_path = code_run_path + 'log_space/'
    model_names = ['lstm','transformer','gru']
    for model_name in model_names:
        get_best_experiments(experiments_res_path,model_name)
