from flask import current_app
 
import os
import json

import numpy as np
import pandas as pd
 
import time
import psutil

from functions.general import get_current_experiment_number
from functions.storage import store_metrics_and_resources
from functions.data import data_augmented_sample

# Refactored and works
def central_worker_data_split(
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

    if central_status['data-split']:
        return False
    
    parameter_folder_path = storage_folder_path + '/parameters/experiment_' + str(current_experiment_number)
    central_parameters_path = parameter_folder_path + '/central.txt'
    central_parameters = None
    with open(central_parameters_path, 'r') as f:
        central_parameters = json.load(f)

    worker_parameters_path = parameter_folder_path + '/worker.txt'
    worker_parameters = None
    with open(worker_parameters_path, 'r') as f:
        worker_parameters = json.load(f)
    
    data_experiment_folder = storage_folder_path + '/data/experiment_' + str(current_experiment_number)
    source_data_path = data_experiment_folder + '/source.csv'
    central_pool_path = data_experiment_folder + '/central_pool.csv'
    worker_pool_path = data_experiment_folder + '/worker_pool.csv'

    os.environ['STATUS'] = 'data splitting'
    
    source_df = pd.read_csv(source_data_path)
    splitted_data_df = source_df.drop('step', axis = 1)
    
    central_data_pool = splitted_data_df.sample(n = central_parameters['sample-pool'])
    central_indexes = central_data_pool.index.tolist()
    splitted_data_df.drop(central_indexes)
    worker_data_pool = splitted_data_df.sample(n = worker_parameters['sample-pool'])

    central_data_pool.to_csv(central_pool_path, index = False)    
    worker_data_pool.to_csv(worker_pool_path, index = False)
    
    central_status['data-split'] = True
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
        'name': 'central-worker-data-split',
        'time-seconds': round(time_diff,5),
        'cpu-percentage': round(cpu_diff,5),
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
# Refactored and works
def split_data_between_workers(
    logger: any
) -> bool:
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    storage_folder_path = 'storage'

    current_experiment_number = get_current_experiment_number()
    status_folder_path = storage_folder_path + '/status/experiment_' + str(current_experiment_number)
    central_status_path = status_folder_path + '/central.txt'
    if not os.path.exists(central_status_path):
        return False
    
    central_status = None
    with open(central_status_path, 'r') as f:
        central_status = json.load(f)

    if not central_status['start']:
        return False

    if central_status['complete']:
        return False

    if not central_status['preprocessed']:
        return False
    
    if central_status['worker-split']:
        return False
    
    worker_status_path = status_folder_path + '/workers.txt'
    if not os.path.exists(worker_status_path):
        return False

    worker_status = None
    with open(worker_status_path, 'r') as f:
        worker_status = json.load(f)

    parameters_folder_path = storage_folder_path + '/parameters/experiment_' + str(current_experiment_number)
    central_parameters_path = parameters_folder_path + '/central.txt'
    if not os.path.exists(central_parameters_path):
        return False
    
    central_parameters = None
    with open(central_parameters_path, 'r') as f:
        central_parameters = json.load(f)

    worker_parameters_path = parameters_folder_path + '/worker.txt'
    if not os.path.exists(worker_parameters_path):
        return False
    
    worker_parameters = None
    with open(worker_parameters_path, 'r') as f:
        worker_parameters = json.load(f)

    data_experiment_folder = storage_folder_path + '/data/experiment_' + str(current_experiment_number)
    worker_pool_path = data_experiment_folder + '/worker_pool.csv'

    os.environ['STATUS'] = 'worker splitting'
    
    worker_pool_df = pd.read_csv(worker_pool_path)

    available_workers = []
    for worker_key in worker_status.keys():
        worker_metadata = worker_status[worker_key]
        if worker_metadata['status'] == 'waiting':
            available_workers.append(worker_key)

    if not central_parameters['min-update-amount'] <= len(available_workers):
        return False
    # Format for worker data is worker_(id)_(cycle)_(size).csv
    if worker_parameters['data-augmentation']['active']:
        for worker_key in available_workers:
            worker_sample_df = data_augmented_sample(
                pool_df = worker_pool_df,
                sample_pool = worker_parameters['data-augmentation']['sample-pool'],
                ratio = worker_parameters['data-augmentation']['1-0-ratio']
            )
            sample_size = worker_sample_df.shape[0]
            data_path = data_experiment_folder + '/worker_' + worker_key + '_' + str(central_status['cycle']) + '_' + str(sample_size) + '.csv'
            worker_sample_df.to_csv(data_path, index = False)
    else:
        worker_df = worker_pool_df.sample(frac = 1)
        worker_dfs = np.array_split(worker_df, len(available_workers))
        index = 0
        for worker_key in available_workers:
            sample_size = worker_dfs[index].shape[0]
            data_path = data_experiment_folder + '/worker_' + worker_key + '_' + str(central_status['cycle']) + '_' + str(sample_size) + '.csv'
            worker_dfs[index].to_csv(data_path, index = False)
            index = index + 1

    central_status['worker-split'] = True
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
        'name': 'split-data-between-workers',
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