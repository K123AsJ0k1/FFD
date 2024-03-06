from flask import current_app

import torch 
import os
import json
import copy
import pandas as pd
import psutil

from collections import OrderedDict

from functions.general import get_current_experiment_number

# Refactored and works
def store_training_context(
    parameters: any,
    df_data: list,
    df_columns: list
) -> bool:
    # Separate training artifacts will have the following folder format of experiment_(int)
    storage_folder_path = 'storage'
    current_experiment_number = get_current_experiment_number()
    central_status_path = storage_folder_path + '/status/experiment_' + str(current_experiment_number) + '/central.txt'
    if os.path.exists(central_status_path):
        central_status = None
        with open(central_status_path, 'r') as f:
            central_status = json.load(f)
        if not central_status['complete']:
            return False

    new_folder_id = current_experiment_number + 1

    template_paths = [
        'storage/parameters/templates/model.txt',
        'storage/parameters/templates/central.txt',
        'storage/parameters/templates/worker.txt',
        'storage/status/templates/central.txt',
        'storage/status/templates/workers.txt',
        'storage/metrics/templates/global.txt',
        'storage/metrics/templates/local.txt',
        'storage/resources/templates/central.txt',
        'storage/resources/templates/workers.txt'
    ]

    for path in template_paths:
        first_split = path.split('.')
        second_split = first_split[0].split('/')

        stored_template = None
        with open(path, 'r') as f:
            stored_template = json.load(f)
        
        file_path = storage_folder_path + '/' + second_split[1] + '/experiment_' + str(new_folder_id) + '/' + second_split[3] + '.txt'
        if second_split[1] == 'parameters':
            given_parameters = parameters[second_split[3]]
            modified_template = copy.deepcopy(stored_template)
            for key in stored_template.keys():
                modified_template[key] = given_parameters[key]
            stored_template = copy.deepcopy(modified_template)
        if second_split[1] == 'resources' and second_split[3] == 'central':
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
        if (second_split[1] == 'metrics' and (second_split[3] == 'global' or second_split[3] == 'local') 
            or (second_split[1] == 'status' and second_split[3] == 'workers')
            or (second_split[1] == 'resources' and second_split[3] == 'workers')):
            stored_template = {}
        
        if not os.path.exists(file_path):
            directory_path = storage_folder_path + '/' + second_split[1] + '/experiment_' + str(new_folder_id)
            os.makedirs(directory_path, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(stored_template, f, indent=4)

    file_path = storage_folder_path + '/data/experiment_' + str(new_folder_id) + '/source.csv'
    if not os.path.exists(file_path):
        directory_path = storage_folder_path + '/data/experiment_' + str(new_folder_id)
        os.makedirs(directory_path, exist_ok=True)
        source_df = pd.DataFrame(df_data, columns = df_columns)
        source_df.to_csv(file_path)
    return True
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
        if subject == 'global':
            data_path = storage_folder_path + '/metrics/experiment_' + str(current_experiment_number) + '/global.txt'
            if not os.path.exists(data_path):
                return False
        
            stored_data = None
            with open(data_path, 'r') as f:
                stored_data = json.load(f)

            new_key = len(stored_data) + 1
            stored_data[str(new_key)] = metrics
    if type == 'resources':
        central_status_path = storage_folder_path + '/status/experiment_' + str(current_experiment_number) + '/central.txt'
        if not os.path.exists(central_status_path):
            return False
        
        central_status = None
        with open(central_status_path, 'r') as f:
            central_status = json.load(f)

        if subject == 'central':
            data_path = storage_folder_path + '/resources/experiment_' + str(current_experiment_number) + '/central.txt'
            if not os.path.exists(data_path):
                return False
            
            stored_data = None
            with open(data_path, 'r') as f:
                stored_data = json.load(f)

            if not str(central_status['cycle']) in stored_data[area]:
                stored_data[area][str(central_status['cycle'])] = {}
            new_key = len(stored_data[area][str(central_status['cycle'])]) + 1
            stored_data[area][str(central_status['cycle'])][str(new_key)] = metrics
    
    with open(data_path, 'w') as f:
        json.dump(stored_data, f, indent=4) 
    
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
        worker_metadata = worker_status[str(status['id'])]
        action = ''
        if worker_metadata['worker-address'] == 'http://' + address + ':7500':
            # When worker is already registered and address has stayed the same
            worker_status[str(status['id'])] = status
            local_metrics[str(status['id'])] = metrics['local']
            workers_resources[str(status['id'])] = metrics['resources']
            action = 'checked'
        else:
            # When worker id has stayed the same, but address has changed due to load balancing
            status['worker-address'] = 'http://' + address + ':7500'
            worker_status[str(status['id'])] = status
            local_metrics[str(status['id'])] = metrics['local']
            workers_resources[str(status['id'])] = metrics['resources']
            action = 'rerouted'
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
# Refactor
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