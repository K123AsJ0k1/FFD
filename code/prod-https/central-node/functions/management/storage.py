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

from prometheus_client import push_to_gateway

from functions.platforms.minio import *
# Refactored
def store_metrics_and_resources( 
   file_lock: any,
   logger: any,
   minio_client: any,
   prometheus_registry: any,
   prometheus_metrics: any,
   type: str,
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

    cycle_folder_path = experiments_folder + '/' + str(central_status['experiment']) + '/' + str(central_status['cycle'])
    object_path = None
    object_data = None
    if type == 'metrics' or type == 'resources':
        if type == 'metrics':
            object_path = cycle_folder_path + '/metrics'
            for key,value in metrics.items():
                if key == 'name':
                    continue
                metric_name = prometheus_metrics['central-global-names'][key]
                prometheus_metrics['central-global'].labels(
                    date = datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
                    experiment = central_status['experiment'], 
                    cycle = central_status['cycle'],
                    metric = metric_name,
                ).set(value)
        if type == 'resources':
            object_path = cycle_folder_path + '/resources/' + area
            action_name = metrics['name']
            for key,value in metrics.items():
                if key == 'name':
                    continue
                metric_name = prometheus_metrics['central-resources-names'][key]
                prometheus_metrics['central-resources'].labels(
                    date = datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
                    experiment = central_status['experiment'], 
                    cycle = central_status['cycle'],
                    area = area,
                    name = action_name,
                    metric = metric_name,
                ).set(value)
            
        wanted_object = get_object_data_and_metadata(
            logger = logger,
            minio_client = minio_client,
            bucket_name = central_bucket,
            object_path = object_path
        )
        if wanted_object is None:
            object_data = {}
        else:
            object_data = wanted_object['data']

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
        #push_to_gateway('http:127.0.0.1:9091', job = 'central-', registry =  prometheus_registry) 
    
    return True
# refactored and works
def store_worker(
    file_lock: any,
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any,
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

    cycle_folder_path = experiments_folder + '/' + str(central_status['experiment']) + '/' + str(central_status['cycle'])
    workers_status_path = cycle_folder_path + '/' + 'workers'
    workers_status_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = workers_status_path
    )
    workers_status = None
    if workers_status_object is None:
        workers_status = {}
    else:
        workers_status = workers_status_object['data']

    used_ids = []
        
    for worker_id in workers_status.keys():
        worker_network_id = workers_status[worker_id]['network-id']
        used_ids.append(int(worker_network_id))
        
    set_of_used_ids = set(used_ids)
    smallest_missing_id = 1
    while smallest_missing_id in set_of_used_ids:
        smallest_missing_id += 1
    # We will assume that MinIO enables memory fault tolerance and logs critical isn't damaged
    info = None
    if not status['worker-id'] in workers_status:
        # When new worker status is started due to experiments
        given_network_id = str(smallest_missing_id)
        given_worker_address = address
        given_experiment = central_status['experiment']
        given_cycle = central_status['cycle']

        status['network-id'] = given_network_id
        status['worker-address'] = given_worker_address
        status['experiment'] = given_experiment
        status['cycle'] = given_cycle
        # Might be anti pattern
        workers_status[status['worker-id']] = status
        info = {
            'message': 'registered', 
            'network-id': given_network_id,
            'worker-address': given_worker_address,
            'experiment': given_experiment,
            'cycle': given_cycle
        }

    else:
        worker_metadata = workers_status[status['worker-id']]
        if worker_metadata['worker-address'] == address:
            # When worker is already registered and address has stayed the same
            workers_status[status['worker-id']] = status
            info = {
                'message': 'checked', 
                'network-id': None,
                'worker-address': None,
                'experiment': None,
                'cycle': None
            }
        else:
            # When worker id has stayed the same, but address has changed due to load balancing
            status['worker-address'] = address
            workers_status[status['worker-id']] = status
            info = {
                'message': 'rerouted', 
                'network-id': None,
                'worker-address': address,
                'experiment': None,
                'cycle': None
            }
    
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = workers_status_path,
        data = workers_status,
        metadata = {}
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
        'name': 'store-worker-' + str(smallest_missing_id),
        'time-seconds': round(time_diff,5),
        'cpu-percentage': cpu_diff,
        'ram-bytes': round(mem_diff,5),
        'disk-bytes': round(disk_diff,5)
    }

    store_metrics_and_resources(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        type = 'resources',
        area = 'function',
        metrics = resource_metrics
    )
    
    return info 
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