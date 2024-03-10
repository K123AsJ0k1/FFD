import pandas as pd
import torch  
import os
import json
import psutil
from pathlib import Path

from collections import OrderedDict

from functions.general import get_current_experiment_number, get_file_data

def store_file_data(
    file_lock: any,
    replace: bool,
    file_folder_path: str,
    file_path: str,
    data: any
):  
    storage_folder_path = 'storage'
    perform = True
    if replace:
        perform = True
    used_folder_path = storage_folder_path + '/' + file_folder_path
    used_file_path = storage_folder_path + '/' + file_path
    relative_path = Path(used_file_path)
    if not replace and relative_path.exists():
        perform = False
    
    if perform:
        with file_lock:
            if not file_folder_path == '':
                os.makedirs(used_folder_path, exist_ok=True)
            if '.txt' in used_file_path:
                with open(used_file_path, 'w') as f:
                    json.dump(data , f, indent=4) 
            if '.csv' in used_file_path:
                data.to_csv(used_file_path, index = False)
            if '.pt' in used_file_path:
                torch.save(data, used_file_path)
# Created and works
def store_central_address(
    file_lock: str,
    central_address: str
) -> bool:
    worker_status_path = 'status/templates/worker.txt'
    worker_status = get_file_data(
        file_lock = file_lock,
        file_path = worker_status_path
    )

    if worker_status is None:
        return False
    
    worker_status['central-address'] = central_address
    store_file_data(
        file_lock = file_lock,
        replace = True,
        file_folder_path = '',
        file_path = worker_status_path,
        data = worker_status 
    )

    return True
# Refactored and works
def store_training_context(
    file_lock: any,
    parameters: any,
    global_model: any,
    df_data: list,
    df_columns: list
) -> any:
    # Separate training artifacts will have the following folder format of experiment_(int)
    current_experiment_number = get_current_experiment_number()
    worker_status_path = 'status/experiment_' + str(current_experiment_number) + '/worker.txt'
    
    worker_status = get_file_data(
        file_lock = file_lock,
        file_path = worker_status_path
    )

    if worker_status is None:
        return {'message': 'no status'}

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
        parameters_folder_path = 'parameters/experiment_' + str(current_experiment_number)
    
        model_parameters_path = parameters_folder_path + '/model.txt'
        worker_parameters_path = parameters_folder_path + '/worker.txt'

        store_file_data(
            file_lock = file_lock,
            replace = True,
            file_folder_path = parameters_folder_path,
            file_path = model_parameters_path,
            data = parameters['model']
        )

        store_file_data(
            file_lock = file_lock,
            replace = True,
            file_folder_path = parameters_folder_path,
            file_path = worker_parameters_path,
            data = parameters['worker']
        )

        worker_status['preprocessed'] = False
        worker_status['trained'] = False
        worker_status['updated'] = False
        worker_status['complete'] = False
        worker_status['cycle'] = parameters['cycle']

    os.environ['STATUS'] = 'storing'
    
    model_folder_path = 'models/experiment_' + str(current_experiment_number)
    global_model_path = model_folder_path + '/global_' + str(worker_status['cycle']-1) + '.pth'
    
    weights = global_model['weights']
    bias = global_model['bias']
    
    formated_parameters = OrderedDict([
        ('linear.weight', torch.tensor(weights,dtype=torch.float32)),
        ('linear.bias', torch.tensor(bias,dtype=torch.float32))
    ])
    
    store_file_data(
        file_lock = file_lock,
        replace = False,
        file_folder_path = model_folder_path,
        file_path = global_model_path,
        data = formated_parameters
    )

    if not df_data == None:
        data_folder_path = 'data/experiment_' + str(current_experiment_number)
        data_path = data_folder_path + '/sample_' + str(worker_status['cycle']) + '.csv'
        worker_df = pd.DataFrame(df_data, columns = df_columns)
        store_file_data(
            file_lock = file_lock,
            replace = False,
            file_folder_path = data_folder_path,
            file_path = data_path,
            data = worker_df
        )
        worker_status['preprocessed'] = False
    
    worker_status['stored'] = True
    store_file_data(
        file_lock = file_lock,
        replace = True,
        file_folder_path = '',
        file_path = worker_status_path,
        data = worker_status
    )

    os.environ['STATUS'] = 'stored'

    return {'message': 'stored'}
# Refactored and works
def store_metrics_and_resources( 
    file_lock: any,
    type: str,
    subject: str,
    area: str,
    metrics: any
) -> bool:
    current_experiment_number = get_current_experiment_number()
    stored_data = None
    data_folder_path = None
    data_file_path = None
    if type == 'metrics':
        if subject == 'local':
            data_folder_path = 'metrics/experiment_' + str(current_experiment_number)
            data_file_path = data_folder_path + '/local.txt'

            stored_data = get_file_data(
                file_lock = file_lock,
                file_path = data_file_path
            )

            if stored_data is None:
                stored_data = {}

            new_key = len(stored_data) + 1
            stored_data[str(new_key)] = metrics
    if type == 'resources':
        current_experiment_number = get_current_experiment_number()
        worker_status_path = 'status/experiment_' + str(current_experiment_number) + '/worker.txt'
        
        worker_status = get_file_data(
            file_lock = file_lock,
            file_path = worker_status_path
        )

        if worker_status is None:
            return False
        
        if subject == 'worker':
            data_folder_path = 'resources/experiment_' + str(current_experiment_number)
            data_file_path = data_folder_path + '/worker.txt'
            
            stored_data = get_file_data(
                file_lock = file_lock,
                file_path = data_file_path
            )

            if stored_data is None:
                stored_data = {
                    'general': {
                        'physical-cpu-amount': psutil.cpu_count(logical=False),
                        'total-cpu-amount': psutil.cpu_count(logical=True),
                        'min-cpu-frequency-mhz': psutil.cpu_freq().min,
                        'max-cpu-frequency-mhz': psutil.cpu_freq().max,
                        'total-ram-amount-bytes': psutil.virtual_memory().total,
                        'available-ram-amount-bytes': psutil.virtual_memory().free,
                        'total-disk-amount-bytes': psutil.disk_usage('.').total,
                        'available-disk-amount-bytes': psutil.disk_usage('.').free
                    },
                    'function': {},
                    'network': {},
                    'training': {},
                    'inference': {}
                }

            if not str(worker_status['cycle']) in stored_data[area]:
                stored_data[area][str(worker_status['cycle'])] = {}
            new_key = len(stored_data[area][str(worker_status['cycle'])]) + 1
            stored_data[area][str(worker_status['cycle'])][str(new_key)] = metrics
    
    store_file_data(
        file_lock = file_lock,
        replace = True,
        file_folder_path = data_folder_path,
        file_path = data_file_path,
        data = stored_data
    ) 
    
    return True