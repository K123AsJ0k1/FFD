from flask import current_app

import torch 
import os
import json
import copy
import pandas as pd
import psutil
import torch

from collections import OrderedDict

from functions.general import get_current_experiment_number, get_file_data
# Created
def store_file_data(
    file_lock: any,
    replace: bool,
    file_folder_path: str,
    file_path: str,
    data: any
):  
    storage_folder_path = 'storage'
    perform = False
    if replace:
        perform = True
    if not replace and not os.path.exists(file_path):
        perform = True
    used_folder_path = storage_folder_path + '/' + file_folder_path
    used_file_path = storage_folder_path + '/' + file_path
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
# Refactored and works
def store_training_context(
    file_lock: any,
    parameters: any,
    df_data: list,
    df_columns: list
) -> bool:
    # Separate training artifacts will have the following folder format of experiment_(int)
    current_experiment_number = get_current_experiment_number()
    central_status_path = 'status/experiment_' + str(current_experiment_number) + '/central.txt'
    if os.path.exists(central_status_path):
        central_status = get_file_data(
            file_lock = file_lock,
            file_path = central_status_path
        )
        if not central_status['complete']:
            return False

    new_folder_id = current_experiment_number + 1

    template_paths = [
        'parameters/templates/model.txt',
        'parameters/templates/central.txt',
        'parameters/templates/worker.txt',
        'status/templates/central.txt',
        'status/templates/workers.txt',
        'metrics/templates/global.txt',
        'metrics/templates/local.txt',
        'resources/templates/central.txt',
        'resources/templates/workers.txt'
    ]

    for template_path in template_paths:
        first_split = template_path.split('.')
        second_split = first_split[0].split('/')

        stored_template = get_file_data(
            file_lock = file_lock,
            file_path = template_path
        )

        modified_template = None
        if second_split[0] == 'parameters':
            given_parameters = parameters[second_split[2]]
            modified_template = copy.deepcopy(stored_template)
            for key in stored_template.keys():
                modified_template[key] = given_parameters[key]
        if second_split[0] == 'status' and second_split[2] == 'central':
            modified_template = copy.deepcopy(stored_template)
        if second_split[0] == 'resources' and second_split[2] == 'central':
            modified_template = {
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
        if (second_split[0] == 'metrics' and (second_split[2] == 'global' or second_split[2] == 'local') 
            or (second_split[0] == 'status' and second_split[2] == 'workers')
            or (second_split[0] == 'resources' and second_split[2] == 'workers')):
            modified_template = {}
        
        modified_template_folder_path = second_split[0] + '/experiment_' + str(new_folder_id)
        modified_template_path = modified_template_folder_path + '/' + second_split[2] + '.txt'
        store_file_data(
            file_lock = file_lock,
            replace = False,
            file_folder_path = modified_template_folder_path,
            file_path =  modified_template_path,
            data = modified_template
        )
    
    data_folder_path = 'data/experiment_' + str(new_folder_id)
    data_path = data_folder_path + '/source.csv'

    source_df = pd.DataFrame(df_data, columns = df_columns)
    store_file_data(
        file_lock = file_lock,
        replace = False,
        file_folder_path = data_folder_path,
        file_path = data_path,
        data = source_df
    )
    
    return True
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
    data_path = None
    if type == 'metrics':
        if subject == 'global':
            data_path = 'metrics/experiment_' + str(current_experiment_number) + '/global.txt'
            stored_data = get_file_data(
                file_lock = file_lock,
                file_path = data_path
            )

            if stored_data is None:
                return False

            new_key = len(stored_data) + 1
            stored_data[str(new_key)] = metrics
    if type == 'resources':
        central_status_path = 'status/experiment_' + str(current_experiment_number) + '/central.txt'
        central_status = get_file_data(
            file_lock = file_lock,
            file_path = central_status_path
        )

        if central_status is None:
            return False
        
        if subject == 'central':
            data_path = 'resources/experiment_' + str(current_experiment_number) + '/central.txt'
        
            stored_data = get_file_data(
                file_lock = file_lock,
                file_path = data_path
            )

            if stored_data is None:
                return False

            if not str(central_status['cycle']) in stored_data[area]:
                stored_data[area][str(central_status['cycle'])] = {}
            new_key = len(stored_data[area][str(central_status['cycle'])]) + 1
            stored_data[area][str(central_status['cycle'])][str(new_key)] = metrics
    
    store_file_data(
        file_lock = file_lock,
        replace = True,
        file_folder_path = '',
        file_path = data_path,
        data = stored_data
    )
    
    return True
# refactored and works
def store_worker(
    address: str,
    status: any,
    metrics: any
) -> any:
    storage_folder_path = 'storage'
    current_experiment_number = get_current_experiment_number()
    status_folder_path = storage_folder_path + '/status/experiment_' + str(current_experiment_number)
    
    central_status_path = status_folder_path + '/central.txt'
    if not os.path.exists(central_status_path):
        return False
    
    central_status = None
    with open(central_status_path, 'r') as f:
        central_status = json.load(f)

    worker_status_path = status_folder_path + '/workers.txt'
    if not os.path.exists(worker_status_path):
        return False
    
    worker_status = None
    # Concurrency issues
    with open(worker_status_path, 'r') as f:
        worker_status = json.load(f)

    local_metrics_path = storage_folder_path + '/metrics/experiment_' + str(current_experiment_number) + '/local.txt'
    if not os.path.exists(local_metrics_path):
        return False
    
    local_metrics = None
    with open(local_metrics_path, 'r') as f:
        local_metrics = json.load(f)

    workers_resources_path = storage_folder_path + '/resources/experiment_' + str(current_experiment_number) + '/workers.txt'
    if not os.path.exists(workers_resources_path):
        return False
    
    workers_resources = None
    with open(workers_resources_path, 'r') as f:
        workers_resources = json.load(f)

    if status['id'] == 0:
        # When worker isn't registered either due to being new or failure restart
        duplicate_id = -1
        used_keys = []
        
        for worker_key in worker_status.keys():
            worker_metadata = worker_status[worker_key]
            if worker_metadata['worker-address'] == status['worker-address']:
                duplicate_id = int(worker_key)
            used_keys.append(int(worker_key))
            
        set_of_used_keys = set(used_keys)
        smallest_missing_id = 1
        while smallest_missing_id in set_of_used_keys:
            smallest_missing_id += 1

        old_local_metrics = {}
        old_resources = {}
        
        if -1 < duplicate_id:
            old_local_metrics = local_metrics[str(duplicate_id)]
            old_resources = workers_resources[str(duplicate_id)]
            del worker_status[str(duplicate_id)]
            del local_metrics[str(duplicate_id)]
            del workers_resources[str(duplicate_id)]

        old_worker_data_path = 'data/experiment_' + str(current_experiment_number) + '/worker_' + str(duplicate_id) + '_' + str(central_status['cycle']) + '.csv'
        if os.path.exists(old_worker_data_path):
            new_worker_data_path = 'worker_' + str(smallest_missing_id) + '_' + str(central_status['cycle']) + '.csv'
            os.rename(old_worker_data_path,new_worker_data_path)

        status['id'] = str(smallest_missing_id)
        status['worker-address'] = 'http://' + address + ':7500'
        status['cycle'] = central_status['cycle']

        worker_status[str(smallest_missing_id)] = status
        local_metrics[str(smallest_missing_id)] = old_local_metrics
        workers_resources[str(smallest_missing_id)] = old_resources

        with open(worker_status_path, 'w') as f:
            json.dump(worker_status, f, indent=4)

        with open(local_metrics_path, 'w') as f:
            json.dump(local_metrics, f, indent=4)

        with open(workers_resources_path, 'w') as f:
            json.dump(workers_resources, f, indent=4)

        existing_metrics = {
            'local': old_local_metrics,
            'resources': old_resources
        }

        payload = {
            'message': 'registered', 
            'experiment_id': current_experiment_number,
            'status': worker_status[str(smallest_missing_id)], 
            'metrics': existing_metrics
        }
        
        return payload 
    else:
        # New experiment has been started
        action = ''
        if not len(worker_status) == 0:
            worker_metadata = worker_status[str(status['id'])]
            if worker_metadata['worker-address'] == 'http://' + address + ':7500':
                # When worker is already registered and address has stayed the same
                if not status is None:
                    worker_status[str(status['id'])] = status
                if not metrics['local'] is None:
                    local_metrics[str(status['id'])] = metrics['local']
                if not metrics['resources'] is None:
                    workers_resources[str(status['id'])] = metrics['resources']
                action = 'checked'
            else:
                # When worker id has stayed the same, but address has changed due to load balancing
                if not status is None:
                    status['worker-address'] = 'http://' + address + ':7500'
                    worker_status[str(status['id'])] = status
                if not metrics['local'] is None:
                    local_metrics[str(status['id'])] = metrics['local']
                if not metrics['resources'] is None:
                    workers_resources[str(status['id'])] = metrics['resources']
                action = 'rerouted'
        else:
            action = 'experiment'
        # status key order doesn't stay the same
        with open(worker_status_path, 'w') as f:
            json.dump(worker_status, f, indent=4)
            
        with open(local_metrics_path, 'w') as f:
            json.dump(local_metrics, f, indent=4)

        with open(workers_resources_path, 'w') as f:
            json.dump(workers_resources, f, indent=4)

        payload = {
            'message': action, 
            'experiment_id': current_experiment_number,
            'status': worker_status, 
            'metrics': None
        }

        return payload
# Refactored and works
def store_update( 
    id: str,
    model: any,
    cycle: int
) -> bool:
    storage_folder_path = 'storage'
    current_experiment_number = get_current_experiment_number()
    status_folder_path = storage_folder_path + '/status/experiment_' + str(current_experiment_number)

    central_status_path = status_folder_path + '/central.txt'
    if not os.path.exists(central_status_path):
        return False
    
    central_status = None
    with open(central_status_path, 'r') as f:
        central_status = json.load(f)

    if not central_status['cycle'] == cycle:
        return False 

    if not central_status['start']:
        return False

    if central_status['complete']:
        return False

    if not central_status['sent']:
        return False
    
    workers_status_path = status_folder_path + '/workers.txt'
    if not os.path.exists(workers_status_path):
        return False

    workers_status = None
    with open(workers_status_path, 'r') as f:
        workers_status = json.load(f)

    # Model format is local_(worker id)_(cycle)_(train_amount).pth
    local_model_folder_path = storage_folder_path + '/models/experiment_' + str(current_experiment_number)
    train_amount = workers_status[id]['train-amount']
    local_model_path = local_model_folder_path + '/local_'  + str(id) + '_' + str(central_status['cycle']) + '_' + str(train_amount) + '.pth'
    
    formatted_model = OrderedDict([
        ('linear.weight', torch.tensor(model['weights'],dtype=torch.float32)),
        ('linear.bias', torch.tensor(model['bias'],dtype=torch.float32))
    ])
    
    torch.save(formatted_model, local_model_path)
    
    workers_status[id]['status'] = 'complete'
    with open(workers_status_path, 'w') as f:
        json.dump(workers_status, f, indent=4) 

    central_status['worker-updates'] = central_status['worker-updates'] + 1
    with open(central_status_path, 'w') as f:
        json.dump(central_status, f, indent=4) 

    return True