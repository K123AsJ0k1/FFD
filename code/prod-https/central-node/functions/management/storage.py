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
# Refactored and works
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
        'name': 'store-worker-' + status['worker-id'],
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
# Refactored
def store_update( 
    file_lock: any,
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any,
    worker_id: str,
    model: any,
    experiment: int,
    cycle: int
) -> bool:
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
    
    experiment_folder_path = experiments_folder + '/' + str(central_status['experiment'])
    cycle_folder_path = experiment_folder_path + '/' + str(central_status['cycle'])
    
    workers_status_path = cycle_folder_path + '/' + 'workers'
    
    workers_status_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = workers_status_path
    )

    if workers_status_object is None:
        return False
    workers_status = workers_status_object['data']

    local_models_folder_path = cycle_folder_path + '/local-models'
    local_model_path = local_models_folder_path + '/' + str(worker_id)
    # Model format is local_(worker id)_(cycle)_(train_amount).pth
    train_amount = workers_status[str(worker_id)]['train-amount']
    test_amount = workers_status[str(worker_id)]['test-amount']
    eval_amount = workers_status[str(worker_id)]['eval-amount']
    
    model_data = OrderedDict([
        ('linear.weight', torch.tensor(model['weights'],dtype=torch.float32)),
        ('linear.bias', torch.tensor(model['bias'],dtype=torch.float32))
    ])
    model_metadata = {
        'train-amount': str(train_amount),
        'test-amount':  str(test_amount),
        'eval-amount':  str(eval_amount),
    }

    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = local_model_path,
        data = model_data,
        metadata = model_metadata
    )
    
    workers_status[id]['status'] = 'complete'
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = workers_status_path,
        data = workers_status,
        metadata = {}
    )

    central_status['worker-updates'] = central_status['worker-updates'] + 1
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path,
        data = central_status,
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
        'name': 'update-from-worker-' + str(id),
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
    
    return True