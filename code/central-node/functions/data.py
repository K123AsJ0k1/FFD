from flask import current_app
 
import os
import json

import numpy as np
import pandas as pd
import torch 
 
import time
import psutil
 
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from functions.general import get_current_experiment_number
from functions.storage import store_metrics_and_resources
# Created and works
def data_augmented_sample(
    pool_df: any,
    sample_pool: int,
    ratio: int
) -> any:
    fraud_cases = pool_df[pool_df['isFraud'] == 1]
    non_fraud_cases = pool_df[pool_df['isFraud'] == 0]

    wanted_fraud_amount = int(sample_pool * ratio)
    wanted_non_fraud_amount = sample_pool-wanted_fraud_amount

    frauds_df = fraud_cases.sample(n = wanted_fraud_amount, replace = True)
    non_fraud_df = non_fraud_cases.sample(n = wanted_non_fraud_amount, replace = True)

    augmented_sample_df = pd.concat([frauds_df,non_fraud_df])
    randomized_sample_df = augmented_sample_df.sample(frac = 1, replace = False)
    return randomized_sample_df
# Refactored and works
def preprocess_into_train_test_and_evaluate_tensors(
    logger: any
) -> bool:
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    storage_folder_path = 'storage'

    current_experiment_number = get_current_experiment_number()
    central_status_path = storage_folder_path + '/status/experiment_' + str(current_experiment_number) + '/central.txt'
    if not os.path.exists(central_status_path):
        return False
    
    central_status = None
    with open(central_status_path, 'r') as f:
        central_status = json.load(f)

    if not central_status['start']:
        return False

    if central_status['complete']:
        return False

    if not central_status['data-split']:
        return False

    if central_status['preprocessed']:
        return False
    
    os.environ['STATUS'] = 'preprocessing'

    parameter_folder_path = storage_folder_path + '/parameters/experiment_' + str(current_experiment_number)
    central_parameters_path = parameter_folder_path + '/central.txt'
    central_parameters = None
    with open(central_parameters_path, 'r') as f:
        central_parameters = json.load(f)

    model_parameters_path = parameter_folder_path + '/model.txt'
    model_parameters = None
    with open(model_parameters_path, 'r') as f:
        model_parameters = json.load(f)

    central_pool_path= storage_folder_path + '/data/experiment_' + str(current_experiment_number) + '/central_pool.csv'
    # tensors have format train/test/eval_(cycle)
    tensor_experiment_folder = storage_folder_path + '/tensors/experiment_' + str(current_experiment_number)
    os.makedirs(tensor_experiment_folder, exist_ok = True)
    train_tensor_path = tensor_experiment_folder + '/train_0.pt'
    test_tensor_path = tensor_experiment_folder + '/test_0.pt'
    eval_tensor_path = tensor_experiment_folder + '/eval_0.pt'

    central_data_df = pd.read_csv(central_pool_path)

    preprocessed_df = None
    if central_parameters['data-augmentation']['active']:
        used_data_df = data_augmented_sample(
            pool_df = central_data_df,
            sample_pool = central_parameters['data-augmentation']['sample-pool'],
            ratio = central_parameters['data-augmentation']['1-0-ratio']
        )
        preprocessed_df = used_data_df[model_parameters['used-columns']]
    else:
        preprocessed_df = central_data_df[model_parameters['used-columns']]
    
    for column in model_parameters['scaled-columns']:
        mean = preprocessed_df[column].mean()
        std_dev = preprocessed_df[column].std()
        preprocessed_df[column] = (preprocessed_df[column] - mean)/std_dev

    X = preprocessed_df.drop(model_parameters['target-column'], axis = 1).values
    y = preprocessed_df[model_parameters['target-column']].values
        
    X_eval, X_train_test, y_eval, y_train_test = train_test_split(
        X, 
        y, 
        train_size = central_parameters['eval-ratio'], 
        random_state = model_parameters['seed']
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_test, 
        y_train_test, 
        train_size = central_parameters['train-ratio'], 
        random_state = model_parameters['seed']
    )

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    X_eval = np.array(X_eval, dtype=np.float32)

    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)
    y_eval = np.array(y_eval, dtype=np.int32)
    
    train_tensor = TensorDataset(
        torch.tensor(X_train), 
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_tensor = TensorDataset(
        torch.tensor(X_test), 
        torch.tensor(y_test, dtype=torch.float32)
    )
    eval_tensor = TensorDataset(
        torch.tensor(X_eval), 
        torch.tensor(y_eval, dtype=torch.float32)
    )

    torch.save(train_tensor,train_tensor_path)
    torch.save(test_tensor,test_tensor_path)
    torch.save(eval_tensor,eval_tensor_path)

    central_status['preprocessed'] = True
    central_status['train-amount'] = X_train.shape[0]
    central_status['test-amount'] = X_test.shape[0]
    central_status['eval-amount'] = X_eval.shape[0]
    with open(central_status_path, 'w') as f:
        json.dump(central_status, f, indent=4) 

    time_end = time.time()
    cpu_end = this_process.cpu_percent(interval=0.2)
    mem_end = psutil.virtual_memory().used 
    disk_end = psutil.disk_usage('.').used
    
    time_diff = (time_end - time_start) 
    cpu_diff = cpu_end - cpu_start 
    mem_diff = (mem_end - mem_start) / (1024 ** 2) 
    disk_diff = (disk_end - disk_start) / (1024 ** 2)

    resource_metrics = {
        'name': 'preprocess-into-train-test-and-evalute-tensors',
        'time-seconds': round(time_diff,5),
        'cpu-percentage': cpu_diff,
        'ram-megabytes': round(mem_diff,5),
        'disk-megabytes': round(disk_diff,5)
    }

    status = store_metrics_and_resources(
        type = 'resources',
        subject = 'central',
        area = 'function',
        metrics = resource_metrics
    )
    
    return True