from flask import current_app

import pandas as pd
import torch  
import os
import json
import psutil

from collections import OrderedDict

from functions.general import get_current_experiment_number
# Created and works
def store_central_address(
    central_address: str
) -> bool:
    storage_folder_path = 'storage'
    
    worker_status_path = storage_folder_path + '/status/templates/worker.txt'
    if not os.path.exists(worker_status_path):
        return False
    
    worker_status = None
    with open(worker_status_path, 'r') as f:
        worker_status = json.load(f)
    
    worker_status['central-address'] = central_address
    with open(worker_status_path, 'w') as f:
        json.dump(worker_status, f, indent=4)

    return True
# Refactored and works
def store_training_context(
    parameters: any,
    global_model: any,
    df_data: list,
    df_columns: list
) -> any:
    storage_folder_path = 'storage'
    # Separate training artifacts will have the following folder format of experiment_(int)
    current_experiment_number = get_current_experiment_number()
    worker_status_path = storage_folder_path + '/status/experiment_' + str(current_experiment_number) + '/worker.txt'
    if not os.path.exists(worker_status_path):
        return {'message': 'no status'}
    
    worker_status = None
    with open(worker_status_path, 'r') as f:
        worker_status = json.load(f)
    
    if worker_status['complete']:
        return {'message': 'complete'}
    
    if not parameters['id'] == worker_status['id']:
        return {'message': 'wrong id'}
    
    if worker_status['stored'] and not worker_status['updated']:
        return {'message': 'ongoing jobs'}
    
    if parameters['model'] == None:
        worker_status['complete'] = True
        worker_status['cycle'] = parameters['cycle']
    else:
        parameters_folder_path = storage_folder_path + '/parameters/experiment_' + str(current_experiment_number)
    
        os.makedirs(parameters_folder_path,exist_ok=True)

        model_parameters_path = parameters_folder_path + '/model.txt'
        worker_parameters_path = parameters_folder_path + '/worker.txt'

        with open(model_parameters_path, 'w') as f:
            json.dump(parameters['model'], f, indent=4)

        with open(worker_parameters_path, 'w') as f:
            json.dump(parameters['worker'], f, indent=4)

        worker_status['preprocessed'] = False
        worker_status['trained'] = False
        worker_status['updated'] = False
        worker_status['complete'] = False
        worker_status['cycle'] = parameters['cycle']

    os.environ['STATUS'] = 'storing'
    
    model_folder_path = storage_folder_path + '/models/experiment_' + str(current_experiment_number)
    os.makedirs(model_folder_path, exist_ok=True)
    global_model_path = model_folder_path + '/global_' + str(worker_status['cycle']-1) + '.pth'
    
    weights = global_model['weights']
    bias = global_model['bias']
    
    formated_parameters = OrderedDict([
        ('linear.weight', torch.tensor(weights,dtype=torch.float32)),
        ('linear.bias', torch.tensor(bias,dtype=torch.float32))
    ])
    
    torch.save(formated_parameters, global_model_path)
    if not df_data == None:
        data_folder_path = storage_folder_path + '/data/experiment_' + str(current_experiment_number)
        os.makedirs(data_folder_path, exist_ok=True)
        worker_data_path = data_folder_path + '/sample_' + str(worker_status['cycle']) + '.csv'
        worker_df = pd.DataFrame(df_data, columns = df_columns)
        worker_df.to_csv(worker_data_path, index = False)
        worker_status['preprocessed'] = False
    
    worker_resource_path = storage_folder_path + '/resources/experiment_' + str(current_experiment_number) + '/worker.txt'
    if not os.path.exists(worker_resource_path):
        stored_template = {
            'general': {
                'physical-cpu-amount': psutil.cpu_count(logical=False),
                'total-cpu-amount': psutil.cpu_count(logical=True),
                'min-cpu-frequency-mhz': psutil.cpu_freq().min,
                'max-cpu-frequency-mhz': psutil.cpu_freq().max,
                'total-ram-amount-megabytes': psutil.virtual_memory().total / (1024 ** 2),
                'available-ram-amount-megabytes': psutil.virtual_memory().free / (1024 ** 2),
                'total-disk-amount-megabytes': psutil.disk_usage('.').total / (1024 ** 2),
                'available-disk-amount-megabytes': psutil.disk_usage('.').free / (1024 ** 2)
            },
            'function': {},
            'network': {},
            'training': {},
            'inference': {}
        }
        with open(worker_resource_path, 'w') as f:
            json.dump(stored_template, f, indent=4)

    worker_status['stored'] = True
    with open(worker_status_path, 'w') as f:
        json.dump(worker_status, f, indent=4)

    os.environ['STATUS'] = 'stored'

    return {'message': 'stored'}
# Refactored and works
def store_metrics_and_resources( 
   type: str,
   subject: str,
   area: str,
   metrics: any
) -> bool:
    storage_folder_path = 'storage'
    current_experiment_number = get_current_experiment_number()
    stored_data = None
    data_path = None
    if type == 'metrics':
        if subject == 'local':
            data_path = storage_folder_path + '/metrics/experiment_' + str(current_experiment_number) + '/local.txt'
            if not os.path.exists(data_path):
                return False
        
            stored_data = None
            with open(data_path, 'r') as f:
                stored_data = json.load(f)

            new_key = len(stored_data) + 1
            stored_data[str(new_key)] = metrics
    if type == 'resources':
        current_experiment_number = get_current_experiment_number()
        worker_status_path = storage_folder_path + '/status/experiment_' + str(current_experiment_number) + '/worker.txt'
        if not os.path.exists(worker_status_path):
            return False
        
        worker_status = None
        with open(worker_status_path, 'r') as f:
            worker_status = json.load(f)

        if subject == 'worker':
            data_path = storage_folder_path + '/resources/experiment_' + str(current_experiment_number) + '/worker.txt'
            if not os.path.exists(data_path):
                return False
            
            stored_data = None
            with open(data_path, 'r') as f:
                stored_data = json.load(f)

            if not str(worker_status['cycle']) in stored_data[area]:
                stored_data[area][str(worker_status['cycle'])] = {}
            new_key = len(stored_data[area][str(worker_status['cycle'])]) + 1
            stored_data[area][str(worker_status['cycle'])][str(new_key)] = metrics
    
    with open(data_path, 'w') as f:
        json.dump(stored_data, f, indent=4) 
    
    return True