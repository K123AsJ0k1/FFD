import torch 
import os
import json
import copy
import pandas as pd
import psutil
import time
import torch
from datetime import datetime
import time

from pathlib import Path
from collections import OrderedDict

from prometheus_client import push_to_gateway
from functions.general import get_experiments_objects, set_experiments_objects

from functions.platforms.minio import *
# Refactored
def store_metrics_resources_and_times( 
   logger: any,
   minio_client: any,
   prometheus_registry: any,
   prometheus_metrics: any,
   type: str,
   area: str,
   metrics: any
) -> bool:
    central_status, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = ''
    )    
    object_name = ''
    replacer = ''
    if type == 'metrics' or type == 'resources' or type == 'times':
        if type == 'metrics':
            object_name = 'metrics'
            source = metrics['name']
            for key,value in metrics.items():
                if key == 'name':
                    continue
                metric_name = prometheus_metrics['global-name'][key]
                prometheus_metrics['global'].labels(
                    date = datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
                    time = time.time(),
                    collector = 'central-main',
                    name = central_status['experiment-name'],
                    experiment = central_status['experiment'], 
                    cycle = central_status['cycle'],
                    source = source,
                    metric = metric_name
                ).set(value)
        if type == 'resources':
            object_name = 'resources'
            replacer = metrics['name']
            set_date = metrics['date']
            set_time = metrics['time']
            source = metrics['name']
            for key,value in metrics.items():
                if key == 'name' or key == 'date' or key == 'time':
                    continue
                metric_name = prometheus_metrics['resource-name'][key]
                prometheus_metrics['resource'].labels(
                    date = set_date,
                    time = set_time,
                    collector = 'central-main', 
                    name = central_status['experiment-name'], 
                    experiment = central_status['experiment'],
                    cycle = central_status['cycle'],
                    source = source,
                    metric = metric_name
                ).set(value)
        if type == 'times':
            object_name = 'action-times'
            replacer = area
            source = metrics['name']
            for key,value in metrics.items():
                if key == 'name':
                    continue
                metric_name = prometheus_metrics['time-name'][key]
                prometheus_metrics['time'].labels(
                    date = datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
                    time = time.time(),
                    collector = 'central-main',
                    name = central_status['experiment-name'],
                    experiment = central_status['experiment'], 
                    cycle = central_status['cycle'],
                    area = area,
                    source = source,
                    metric = metric_name
                ).set(value)

        wanted_data, _ = get_experiments_objects(
            logger = logger,
            minio_client = minio_client,
            object = object_name,
            replacer = replacer
        )
        object_data = None
        if wanted_data is None:
            object_data = {}
        else:
            object_data = wanted_data

        new_key = len(object_data) + 1
        object_data[str(new_key)] = metrics

        set_experiments_objects(
            logger = logger,
            minio_client = minio_client,
            object = object_name,
            replacer = replacer,
            overwrite = True,
            object_data = object_data,
            object_metadata = {}
        )
        #push_to_gateway('http:127.0.0.1:9091', job = 'central-', registry =  prometheus_registry) 
    return True
# refactored and works
def store_worker(
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any,
    address: str,
    status: any
) -> any:
    time_start = time.time()

    central_status, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = ''
    )

    workers_status, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'workers',
        replacer = ''
    )

    if workers_status is None:
        workers_status = {}
    
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
    
    set_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'workers',
        replacer = '',
        overwrite = True,
        object_data = workers_status,
        object_metadata = {}
    )

    time_end = time.time()
    time_diff = (time_end - time_start) 
    resource_metrics = {
        'name': 'store-worker-' + status['worker-id'],
        'action-time-start': time_start,
        'action-time-end': time_end,
        'action-total-seconds': round(time_diff,5),
    }

    store_metrics_resources_and_times(
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        type = 'times',
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
    time_start = time.time()

    central_status, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = ''
    )
    
    if central_status is None:
        return {'message': 'no status'}
    
    if not str(central_status['experiment']) == str(experiment):
        return {'message': 'incorrect experiment'}
    
    if not str(central_status['cycle']) == str(cycle):
        return {'message': 'incorrect cycle'}
    
    if not central_status['start']:
        return {'message': 'no start'}
   
    if central_status['complete']:
        return {'message': 'already complete'}
    
    if not central_status['sent']:
        return {'message': 'not sent'}
    
    workers_status, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'workers',
        replacer = ''
    )

    if workers_status is None:
        return False
    
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
    
    set_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'local-models',
        replacer = str(worker_id),
        overwrite = True,
        object_data = model_data,
        object_metadata = model_metadata
    )
    
    central_status['worker-updates'] = central_status['worker-updates'] + 1
    set_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = '',
        overwrite = True,
        object_data = central_status,
        object_metadata = {}
    )

    time_end = time.time()
    time_diff = (time_end - time_start) 
    resource_metrics = {
        'name': 'update-from-worker-' + str(id),
        'action-time-start': time_start,
        'action-time-end': time_end,
        'action-total-seconds': round(time_diff,5),
    }

    store_metrics_resources_and_times(
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        type = 'times',
        area = 'function',
        metrics = resource_metrics
    )
    
    return {'message': 'stored'}