from flask import current_app

import torch 
import os
import json
import copy
import pandas as pd
import psutil
import time
import torch
from pathlib import Path

from collections import OrderedDict

from functions.general import get_current_experiment_number, get_file_data
# Created and works
def store_file_data(
    file_lock: any,
    replace: bool,
    file_folder_path: str,
    file_path: str,
    data: any
):  
    storage_folder_path = 'storage'
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

        if central_status is None:
            return False

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
    file_lock: any,
    address: str,
    status: any,
    metrics: any
) -> any:
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    current_experiment_number = get_current_experiment_number()
    status_folder_path = 'status/experiment_' + str(current_experiment_number)
    
    central_status_path = status_folder_path + '/central.txt'

    central_status = get_file_data(
        file_lock = file_lock,
        file_path = central_status_path
    )

    if central_status is None:
        return False
    
    workers_status_path = status_folder_path + '/workers.txt'
    workers_status = get_file_data(
        file_lock = file_lock,
        file_path = workers_status_path
    )

    if workers_status is None:
        return False
    
    local_metrics_path =  'metrics/experiment_' + str(current_experiment_number) + '/local.txt'
    local_metrics = get_file_data(
        file_lock = file_lock,
        file_path = local_metrics_path 
    )

    if local_metrics is None:
        return False

    workers_resources_path = 'resources/experiment_' + str(current_experiment_number) + '/workers.txt'

    workers_resources = get_file_data(
        file_lock = file_lock,
        file_path = workers_resources_path 
    )

    if workers_resources is None:
        return False
    
    if status['id'] == 0:
        # When worker isn't registered either due to being new or failure restart
        duplicate_id = -1
        used_keys = []
        
        for worker_key in workers_status.keys():
            worker_metadata = workers_status[worker_key]
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
            del workers_status[str(duplicate_id)]
            del local_metrics[str(duplicate_id)]
            del workers_resources[str(duplicate_id)]

        old_worker_data_path = 'data/experiment_' + str(current_experiment_number) + '/worker_' + str(duplicate_id) + '_' + str(central_status['cycle']) + '.csv'
        if os.path.exists(old_worker_data_path):
            new_worker_data_path = 'worker_' + str(smallest_missing_id) + '_' + str(central_status['cycle']) + '.csv'
            os.rename(old_worker_data_path,new_worker_data_path)

        status['id'] = str(smallest_missing_id)
        status['worker-address'] = 'http://' + address + ':7500'
        status['cycle'] = central_status['cycle']

        workers_status[str(smallest_missing_id)] = status
        local_metrics[str(smallest_missing_id)] = old_local_metrics
        workers_resources[str(smallest_missing_id)] = old_resources

        store_file_data(
            file_lock = file_lock,
            replace = True,
            file_folder_path = '',
            file_path = workers_status_path,
            data = workers_status,
        )

        store_file_data(
            file_lock = file_lock,
            replace = True,
            file_folder_path = '',
            file_path = local_metrics_path,
            data = local_metrics,
        )

        store_file_data(
            file_lock = file_lock,
            replace = True,
            file_folder_path = '',
            file_path = workers_resources_path,
            data = workers_resources,
        )

        existing_metrics = {
            'local': old_local_metrics,
            'resources': old_resources
        }

        payload = {
            'message': 'registered', 
            'experiment_id': current_experiment_number,
            'status': workers_status[str(smallest_missing_id)], 
            'metrics': existing_metrics
        }

        time_end = time.time()
        cpu_end = this_process.cpu_percent(interval=0.2)
        mem_end = psutil.virtual_memory().used 
        disk_end = psutil.disk_usage('.').used

        time_diff = (time_end - time_start) 
        cpu_diff = cpu_end - cpu_start 
        mem_diff = (mem_end - mem_start) / (1024 ** 2) 
        disk_diff = (disk_end - disk_start) / (1024 ** 2)

        resource_metrics = {
            'name': 'store-worker-' + str(smallest_missing_id),
            'time-seconds': round(time_diff,5),
            'cpu-percentage': cpu_diff,
            'ram-megabytes': round(mem_diff,5),
            'disk-megabytes': round(disk_diff,5)
        }

        status = store_metrics_and_resources(
            file_lock = file_lock,
            type = 'resources',
            subject = 'central',
            area = 'function',
            metrics = resource_metrics
        )
        
        return payload 
    else:
        # New experiment has been started
        action = ''
        if not len(workers_status) == 0:
            worker_metadata = workers_status[str(status['id'])]
            if worker_metadata['worker-address'] == 'http://' + address + ':7500':
                # When worker is already registered and address has stayed the same
                if not status is None:
                    workers_status[str(status['id'])] = status
                if not metrics['local'] is None:
                    local_metrics[str(status['id'])] = metrics['local']
                if not metrics['resources'] is None:
                    workers_resources[str(status['id'])] = metrics['resources']
                action = 'checked'
            else:
                # When worker id has stayed the same, but address has changed due to load balancing
                if not status is None:
                    status['worker-address'] = 'http://' + address + ':7500'
                    workers_status[str(status['id'])] = status
                if not metrics['local'] is None:
                    local_metrics[str(status['id'])] = metrics['local']
                if not metrics['resources'] is None:
                    workers_resources[str(status['id'])] = metrics['resources']
                action = 'rerouted'
        else:
            action = 'experiment'
        # status key order doesn't stay the same
        store_file_data(
            file_lock = file_lock,
            replace = True,
            file_folder_path = '',
            file_path = workers_status_path,
            data = workers_status,
        )

        store_file_data(
            file_lock = file_lock,
            replace = True,
            file_folder_path = '',
            file_path = local_metrics_path,
            data = local_metrics,
        )

        store_file_data(
            file_lock = file_lock,
            replace = True,
            file_folder_path = '',
            file_path = workers_resources_path,
            data = workers_resources,
        )

        payload = {
            'message': action, 
            'experiment_id': current_experiment_number,
            'status': workers_status[str(status['id'])], 
            'metrics': None
        }

        time_end = time.time()
        cpu_end = this_process.cpu_percent(interval=0.2)
        mem_end = psutil.virtual_memory().used 
        disk_end = psutil.disk_usage('.').used

        time_diff = (time_end - time_start) 
        cpu_diff = cpu_end - cpu_start 
        mem_diff = (mem_end - mem_start) / (1024 ** 2) 
        disk_diff = (disk_end - disk_start) / (1024 ** 2)

        resource_metrics = {
            'name': 'store-worker-' + str(status['id']),
            'time-seconds': round(time_diff,5),
            'cpu-percentage': cpu_diff,
            'ram-megabytes': round(mem_diff,5),
            'disk-megabytes': round(disk_diff,5)
        }

        status = store_metrics_and_resources(
            file_lock = file_lock,
            type = 'resources',
            subject = 'central',
            area = 'function',
            metrics = resource_metrics
        )

        return payload
# Refactored and works
def store_update( 
    file_lock: any,
    id: str,
    model: any,
    cycle: int
) -> bool:
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    current_experiment_number = get_current_experiment_number()
    status_folder_path = 'status/experiment_' + str(current_experiment_number)

    central_status_path = status_folder_path + '/central.txt'
    central_status = get_file_data(
        file_lock = file_lock,
        file_path = central_status_path
    )

    if central_status is None:
        return False

    if not central_status['cycle'] == cycle:
        return False 

    if not central_status['start']:
        return False

    if central_status['complete']:
        return False

    if not central_status['sent']:
        return False
    
    workers_status_path = status_folder_path + '/workers.txt'
    
    workers_status = get_file_data(
        file_lock = file_lock,
        file_path = workers_status_path
    )

    if workers_status is None:
        return False

    # Model format is local_(worker id)_(cycle)_(train_amount).pth
    local_model_folder_path = 'models/experiment_' + str(current_experiment_number)
    train_amount = workers_status[id]['train-amount']
    local_model_path = local_model_folder_path + '/local_'  + str(id) + '_' + str(central_status['cycle']) + '_' + str(train_amount) + '.pth'
    
    formatted_model = OrderedDict([
        ('linear.weight', torch.tensor(model['weights'],dtype=torch.float32)),
        ('linear.bias', torch.tensor(model['bias'],dtype=torch.float32))
    ])

    store_file_data(
        file_lock = file_lock,
        replace = False,
        file_folder_path = local_model_folder_path,
        file_path = local_model_path,
        data = formatted_model
    )
    
    workers_status[id]['status'] = 'complete'
    store_file_data(
        file_lock = file_lock,
        replace = True,
        file_folder_path = '',
        file_path = workers_status_path,
        data =  workers_status
    )

    central_status['worker-updates'] = central_status['worker-updates'] + 1
    store_file_data(
        file_lock = file_lock,
        replace = True,
        file_folder_path = '',
        file_path = central_status_path,
        data =  central_status
    )

    time_end = time.time()
    cpu_end = this_process.cpu_percent(interval=0.2)
    mem_end = psutil.virtual_memory().used 
    disk_end = psutil.disk_usage('.').used

    time_diff = (time_end - time_start) 
    cpu_diff = cpu_end - cpu_start 
    mem_diff = (mem_end - mem_start) / (1024 ** 2) 
    disk_diff = (disk_end - disk_start) / (1024 ** 2)

    resource_metrics = {
        'name': 'update-from-worker-' + str(id),
        'time-seconds': round(time_diff,5),
        'cpu-percentage': cpu_diff,
        'ram-megabytes': round(mem_diff,5),
        'disk-megabytes': round(disk_diff,5)
    }

    status = store_metrics_and_resources(
        file_lock = file_lock,
        type = 'resources',
        subject = 'central',
        area = 'function',
        metrics = resource_metrics
    )
    
    return True