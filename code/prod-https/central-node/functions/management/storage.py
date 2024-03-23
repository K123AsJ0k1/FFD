import torch 
import os
import json
import copy
import pandas as pd
import psutil
import time
import torch
from datetime import datetime

from pathlib import Path
from collections import OrderedDict

from functions.platforms.minio import *
# Refactored
def store_metrics_and_resources( 
   file_lock: any,
   logger: any,
   minio_client: any,
   type: str,
   subject: str,
   area: str,
   metrics: any
) -> bool:
    experiments_folder = 'experiments'
    central_bucket = 'central'
    central_status_path = experiments_folder + '/status'
    central_status_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path
    )
    central_status = central_status_object['data']

    cycle_folder_path = experiments_folder + '/' + str(central_status['experiments']) + '/' + str(central_status['cycle'])
    object_path = None
    object_data = None
    if type == 'metrics' or type == 'resources':
        if subject == 'global':
            object_path = cycle_folder_path + '/metrics'
        if subject == 'central':
            object_path = cycle_folder_path + '/resources/' + area
    
        wanted_object = get_object_data_and_metadata(
            logger = logger,
            minio_client = minio_client,
            bucket_name = central_bucket,
            object_path = object_path
        )
        object_data = wanted_object['data']

        if object_data is None:
            object_data = {}

        new_key = len(object_data) + 1
        object_data[str(new_key)] = metrics
    
        create_or_update_object(
            logger = logger,
            minio_client = minio_client,
            bucket_name = central_bucket,
            object_path = object_path,
            data = object_data,
            metadata = {}
        )
    
    return True
# refactored
def store_worker(
    file_lock: any,
    logger: any,
    minio_client: any,
    address: str,
    status: any
) -> any:
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    experiments_folder = 'experiments'
    central_bucket = 'central'
    central_status_path = experiments_folder + '/status'
    central_status_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path
    )
    central_status = central_status_object['data']

    cycle_folder_path = experiments_folder + '/' + str(central_status['experiments']) + '/' + str(central_status['cycle'])
    workers_status_path = cycle_folder_path + '/' + 'workers'
    workers_status_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = workers_status_path
    )
    workers_status = workers_status_object['data']

    if workers_status is None:
        workers_status = {}

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

        if -1 < duplicate_id:
            del workers_status[str(duplicate_id)]
            
        old_worker_data_path = cycle_folder_path + '/worker-' + str(duplicate_id)
        old_data_exists = check_object(
            logger = logger,
            minio_client = minio_client,
            bucket_name = central_bucket,
            object_path = old_worker_data_path
        )
        
        if old_data_exists:
            new_worker_data_path = cycle_folder_path + '/worker-' + str(smallest_missing_id) 
            worker_data_object = get_object_data_and_metadata(
                logger = logger,
                minio_client = minio_client,
                bucket_name = central_bucket,
                object_path = old_worker_data_path
            )
        
            create_or_update_object(
                logger = logger,
                minio_client = minio_client,
                bucket_name = central_bucket,
                object_path = new_worker_data_path,
                data = worker_data_object['data'],
                metadata = worker_data_object['metadat']
            )
            
            delete_object(
                logger = logger,
                minio_client = minio_client,
                bucket_name = central_bucket,
                object_path = old_worker_data_path
            )

        status['id'] = str(smallest_missing_id)
        status['worker-address'] = address
        status['cycle'] = central_status['cycle']

        workers_status[str(smallest_missing_id)] = status

        create_or_update_object(
            logger = logger,
            minio_client = minio_client,
            bucket_name = central_bucket,
            object_path = workers_status_path,
            data = workers_status[str(smallest_missing_id)],
            metadata = {}
        )
       
        payload = {
            'message': 'registered', 
            'experiment': central_status['experiments']
        }

        time_end = time.time()
        cpu_end = this_process.cpu_percent(interval=0.2)
        mem_end = psutil.virtual_memory().used 
        disk_end = psutil.disk_usage('.').used

        time_diff = (time_end - time_start) 
        cpu_diff = cpu_end - cpu_start 
        mem_diff = (mem_end - mem_start) 
        disk_diff = (disk_end - disk_start)

        resource_metrics = {
            'name': 'store-worker-' + str(smallest_missing_id),
            'time-seconds': round(time_diff,5),
            'cpu-percentage': cpu_diff,
            'ram-bytes': round(mem_diff,5),
            'disk-bytes': round(disk_diff,5)
        }

        store_metrics_and_resources(
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
            if worker_metadata['worker-address'] == address:
                # When worker is already registered and address has stayed the same
                if not status is None:
                    workers_status[str(status['id'])] = status
                action = 'checked'
            else:
                # When worker id has stayed the same, but address has changed due to load balancing
                if not status is None:
                    status['worker-address'] = 'http://' + address + ':7500'
                    workers_status[str(status['id'])] = status
                action = 'rerouted'
        else:
            action = 'experiment'
        
        create_or_update_object(
            logger = logger,
            minio_client = minio_client,
            bucket_name = central_bucket,
            object_path = workers_status_path,
            data = workers_status[str(smallest_missing_id)],
            metadata = {}
        )

        payload = None
        if not action == 'experiment':
            payload = {
                'message': action, 
                'experiment': central_status['experiments']
            }
        else:
            payload = {
                'message': action, 
                'experiment': central_status['experiments']
            }

        time_end = time.time()
        cpu_end = this_process.cpu_percent(interval=0.2)
        mem_end = psutil.virtual_memory().used 
        disk_end = psutil.disk_usage('.').used

        time_diff = (time_end - time_start) 
        cpu_diff = cpu_end - cpu_start 
        mem_diff = (mem_end - mem_start) 
        disk_diff = (disk_end - disk_start)

        resource_metrics = {
            'name': 'store-worker-' + str(status['id']),
            'time-seconds': round(time_diff,5),
            'cpu-percentage': cpu_diff,
            'ram-bytes': round(mem_diff,5),
            'disk-bytes': round(disk_diff,5)
        }

        store_metrics_and_resources(
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
    mem_diff = (mem_end - mem_start) 
    disk_diff = (disk_end - disk_start)

    resource_metrics = {
        'name': 'update-from-worker-' + str(id),
        'time-seconds': round(time_diff,5),
        'cpu-percentage': cpu_diff,
        'ram-bytes': round(mem_diff,5),
        'disk-bytes': round(disk_diff,5)
    }

    status = store_metrics_and_resources(
        file_lock = file_lock,
        type = 'resources',
        subject = 'central',
        area = 'function',
        metrics = resource_metrics
    )
    
    return True