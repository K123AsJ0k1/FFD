from flask import current_app

import numpy as np
import pandas as pd
import torch  
import os
import json
import time 
import psutil
 
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from functions.general import get_current_experiment_number, get_file_data
from functions.storage import store_metrics_and_resources, store_file_data
# Refactored and works
def preprocess_into_train_test_and_eval_tensors(
    file_lock: any,
    logger: any
) -> bool:
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    current_experiment_number = get_current_experiment_number()
    worker_status_path = 'status/experiment_' + str(current_experiment_number) + '/worker.txt'
    
    worker_status = get_file_data(
        file_lock = file_lock,
        file_path = worker_status_path
    )

    if worker_status is None:
        return False
    
    if worker_status['complete']:
        return False

    if not worker_status['stored']:
        return False

    if worker_status['preprocessed']:
        return False

    os.environ['STATUS'] = 'preprocessing'
    
    parameter_folder_path = 'parameters/experiment_' + str(current_experiment_number)
    model_parameters_path = parameter_folder_path + '/model.txt'
    worker_parameters_path = parameter_folder_path + '/worker.txt'

    model_parameters = get_file_data(
        file_lock = file_lock,
        file_path = model_parameters_path
    )

    if model_parameters is None:
        return False

    worker_parameters = get_file_data(
        file_lock = file_lock,
        file_path = worker_parameters_path
    )

    if worker_parameters is None:
        return False

    worker_data_path = 'data/experiment_' + str(current_experiment_number) + '/sample_' + str(worker_status['cycle']) + '.csv'
    sample_df = get_file_data(
        file_lock = file_lock,
        file_path = worker_data_path
    )

    preprocessed_df = sample_df[model_parameters['used-columns']]
    for column in model_parameters['scaled-columns']:
        mean = preprocessed_df[column].mean()
        std_dev = preprocessed_df[column].std()
        preprocessed_df[column] = (preprocessed_df[column] - mean)/std_dev

    X = preprocessed_df.drop(model_parameters['target-column'], axis = 1).values
    y = preprocessed_df[model_parameters['target-column']].values
        
    X_eval, X_train_test, y_eval, y_train_test = train_test_split(
        X, 
        y, 
        train_size = worker_parameters['eval-ratio'], 
        random_state = model_parameters['seed']
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_test, 
        y_train_test, 
        train_size = worker_parameters['train-ratio'], 
        random_state = model_parameters['seed']
    )

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    X_eval = np.array(X_eval, dtype=np.float32)

    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)
    y_eval = np.array(y_eval, dtype=np.float32)
    
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

    tensor_folder_path = 'tensors/experiment_' + str(current_experiment_number)
    train_tensor_path = tensor_folder_path + '/train_' + str(worker_status['cycle']) + '.pt'
    store_file_data(
        file_lock = file_lock,
        replace = False,
        file_folder_path = tensor_folder_path,
        file_path = train_tensor_path,
        data = train_tensor
    )
    test_tensor_path = tensor_folder_path + '/test_' + str(worker_status['cycle']) + '.pt'
    store_file_data(
        file_lock = file_lock,
        replace = False,
        file_folder_path = tensor_folder_path,
        file_path = test_tensor_path,
        data = test_tensor
    )
    eval_tensor_path = tensor_folder_path + '/eval_' + str(worker_status['cycle']) + '.pt'
    store_file_data(
        file_lock = file_lock,
        replace = False,
        file_folder_path = tensor_folder_path,
        file_path = eval_tensor_path,
        data = eval_tensor
    )
    
    worker_status['preprocessed'] = True
    worker_status['train-amount'] = X_train.shape[0]
    worker_status['test-amount'] = X_test.shape[0]
    worker_status['eval-amount'] = X_eval.shape[0]
    
    store_file_data(
        file_lock = file_lock,
        replace = True,
        file_folder_path = '',
        file_path = worker_status_path,
        data = worker_status
    )
    
    os.environ['STATUS'] = 'preprocessed'

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
        file_lock = file_lock,
        type = 'resources',
        subject = 'worker',
        area = 'function',
        metrics = resource_metrics
    )

    return True 