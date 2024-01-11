import os,sys
import json
from pathlib import Path as path

root_path = '/home/chongz/programFile/ML-Final-Test/ML-Final/'

# define the code running path
code_run_path = root_path

# define the result save path
result_root_path = code_run_path + 'experiment/experiment_analysis/result/'
result_root_path_single = code_run_path + 'experiment/experiment_analysis/result_single/'

# check the result save path
res_path_list = [result_root_path,result_root_path_single]
for res_path in res_path_list:
    if not os.path.exists(path(res_path)):
        os.makedirs(path(res_path))
    assert os.path.exists(path(res_path)), "path not exists"

def get_single_best_experiments(experiments_res_path, mode:str,metric:str):
    """
    this function is to get the single best metric experiment result

    experiment_res_path: the log path
    mode: model name
    metric:one of the [test_MSE,test_MAE,test_MSE_std,test_MAE_std]
    """

    # dump result
    best_metric = {"best_MSE": 10000000000.0,"best_MAE": 10000000000.0}
    test_metric = {"test_MSE": 10000000000.0,"test_MAE": 10000000000.0,"test_MSE_std": 10000000000.0,"test_MAE_std": 10000000000.0}
    hyperparams = {}
    train_output = {}

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
        metric_val,metric__val_tmp = test_metric[metric],test_metric_tmp[metric]

        if metric__val_tmp > metric_val:
            continue
        elif metric__val_tmp < metric_val or hyperparams_tmp["epochs"] < hyperparams["epochs"]:
            test_metric,best_metric,hyperparams,train_output = test_metric_tmp,best_metric_tmp, hyperparams_tmp,train_output_tmp

    agg_res = {
        "test_metric": test_metric,
        "best_metric": best_metric,
        "train_output": train_output,
        "hyperparams": hyperparams
    }
    with open(path(result_root_path_single + mode + '/' + metric + '_'+ outer_folder + '.json'), 'w') as file:
        json.dump(agg_res, file, indent=4)


def get_best_experiments(experiments_res_path, mode:str):
    """
    experiment_res_path: the log path
    mode: model name
    """
    # dump result
    best_metric = {"best_MSE": 10000000000.0,"best_MAE": 10000000000.0}
    test_metric = {"test_MSE": 10000000000.0,"test_MAE": 10000000000.0,"test_MSE_std": 10000000000.0,"test_MAE_std": 10000000000.0}
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
        t_mse_std,t_mae_std,t_mse_t_std,t_mae_t_std = (test_metric['test_MSE_std'],test_metric["test_MAE_std"],
                                      test_metric_tmp['test_MSE_std'],test_metric_tmp['test_MAE_std'] )
        # 如果有任意一个指标比之前的更大
        if (t_mse_t > t_mse or t_mae_t > t_mae or t_mse_t_std > t_mse_std or t_mae_t_std > t_mae_std):
            continue
        #所有的指标都比之前的小，或都相等且epochs更小
        elif (t_mse_t < t_mse or t_mae_t < t_mae or t_mse_t_std < t_mse_std or t_mae_t_std < t_mae_std) or hyperparams_tmp["epochs"] < hyperparams["epochs"]:
            test_metric,best_metric,hyperparams,train_output = test_metric_tmp,best_metric_tmp, hyperparams_tmp,train_output_tmp

    agg_res = {
        "test_metric": test_metric,
        "best_metric": best_metric,
        "train_output": train_output,
        "hyperparams": hyperparams
    }
    with open(path(result_root_path + mode + '_'+ outer_folder + '.json'), 'w') as file:
        json.dump(agg_res, file, indent=4)
            



if __name__ == '__main__':
    os.chdir(code_run_path)
    sys.path.insert(0, code_run_path)
    experiments_res_path = code_run_path + 'log_space/'
    model_names = ['lstm','transformer','gru']
    # metrics = ['test_MSE','test_MAE','test_MSE_std','test_MAE_std']
    metrics = ['test_MSE']

    # create result for every model
    for model_name in model_names:
        tpp = result_root_path_single + model_name + '/'
        if not os.path.exists(path(tpp)):
            os.makedirs(path(tpp))
            assert os.path.exists(path(tpp)), "path not exists"

    # compare all metrics, then get the experiment result
    # for model_name in model_names:
    #     get_best_experiments(experiments_res_path,model_name)
    
    # compare single metric, then get the experiment result
    for model_name in model_names:
        for metric in metrics:
            get_single_best_experiments(experiments_res_path,model_name,metric)
    
