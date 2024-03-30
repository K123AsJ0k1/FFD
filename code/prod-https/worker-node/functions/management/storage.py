import pandas as pd
import torch  
import os
import json
import psutil
from pathlib import Path
from datetime import datetime
import time

from collections import OrderedDict

from functions.platforms.minio import get_object_data_and_metadata, create_or_update_object, check_object
from functions.general import encode_metadata_lists_to_strings
from functions.platforms.mlflow import start_experiment, check_experiment
# Refactored and works
def store_training_context(
    file_lock: any,
    logger: any,
    minio_client: any,
    mlflow_client: any,
    prometheus_registry: any,
    prometheus_metrics: any,
    info: any,
    global_model: any,
    df_data: list,
    df_columns: list
) -> any:
    workers_bucket = 'workers'
    worker_experiments_folder = os.environ.get('WORKER_ID') + '/experiments'
    worker_status_path = worker_experiments_folder + '/status'
    worker_status_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = workers_bucket,
        object_path = worker_status_path
    )
    worker_status = worker_status_object['data']

    if worker_status is None:
        return {'message': 'no status'}

    if worker_status['complete']:
        return {'message': 'complete'}
    
    if not info['worker-id'] == worker_status['worker-id']:
        return {'message': 'incorrect'}
    
    if worker_status['stored'] and not worker_status['updated']:
        return {'message': 'ongoing'}
    
    os.environ['STATUS'] = 'storing'

    experiment_folder_path = worker_experiments_folder + '/' + str(info['experiment'])
    cycle_folder_path = experiment_folder_path + '/' + str(info['cycle'])
    parameters_folder_path = experiment_folder_path + '/parameters'
    times_path = experiment_folder_path + '/times'
    global_model_path = cycle_folder_path + '/global-model'
    if info['model'] == None:
        worker_status['complete'] = True
        worker_status['cycle'] = info['cycle']

        times_object = get_object_data_and_metadata(
            logger = logger,
            minio_client = minio_client,
            bucket_name = workers_bucket,
            object_path = times_path
        )
        times = times_object['data']

        experiment_start = times['experiment-time-start']
        experiment_end = time.time()
        experiment_total = experiment_end - experiment_start
        times['experiment-time-end'] = experiment_end
        times['experiment-total-seconds'] = experiment_total
        create_or_update_object(
            logger = logger,
            minio_client = minio_client,
            bucket_name = workers_bucket,
            object_path = times_path,
            data = times,
            metadata = {}
        )
    else:
        object_exists = check_object(
            logger = logger,
            minio_client = minio_client,
            bucket_name = workers_bucket,
            object_path = times_path
        )
        if not object_exists:
            times = {
                'cycle': str(info['cycle']),
                'experiment-date': datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
                'experiment-time-start': time.time(),
                'experiment-time-end':0,
                'experiment-total-seconds': 0,
            }
            create_or_update_object(
                logger = logger,
                minio_client = minio_client,
                bucket_name = workers_bucket,
                object_path = times_path,
                data = times,
                metadata = {}
            )

        worker_experiment_name = 'worker-' + str(os.environ.get('WORKER_ID')) + '-' + str(info['experiment'])
        experiment_dict = check_experiment(
            logger = logger,
            mlflow_client = mlflow_client,
            experiment_name = worker_experiment_name
        )
        if experiment_dict is None:
            experiment_id = start_experiment(
                logger = logger,
                mlflow_client = mlflow_client,
                experiment_name = worker_experiment_name,
                experiment_tags = {}
            )
            worker_status['experiment-id'] = experiment_id

        model_parameters_path = parameters_folder_path + '/model'
        object_exists = check_object(
            logger = logger,
            minio_client = minio_client,
            bucket_name = workers_bucket,
            object_path = model_parameters_path
        )

        if not object_exists:
            create_or_update_object(
                logger = logger,
                minio_client = minio_client,
                bucket_name = workers_bucket,
                object_path = model_parameters_path,
                data = info['model'],
                metadata = {}
            )

        worker_parameters_path = parameters_folder_path + '/worker'
        object_exists = check_object(
            logger = logger,
            minio_client = minio_client,
            bucket_name = workers_bucket,
            object_path = worker_parameters_path
        )

        if not object_exists:
            create_or_update_object(
                logger = logger,
                minio_client = minio_client,
                bucket_name = workers_bucket,
                object_path = worker_parameters_path,
                data = info['worker'],
                metadata = {}
            )

        data_path = cycle_folder_path + '/worker-sample'
        object_exists = check_object(
            logger = logger,
            minio_client = minio_client,
            bucket_name = workers_bucket,
            object_path = data_path 
        )

        if not object_exists:
            metadata = encode_metadata_lists_to_strings({
                'header': df_columns,
                'columns': len(df_columns),
                'rows': len(df_data)
            })
            create_or_update_object(
                logger = logger,
                minio_client = minio_client,
                bucket_name = workers_bucket,
                object_path = data_path,
                data = df_data,
                metadata = metadata
            )

        worker_status['preprocessed'] = False
        worker_status['trained'] = False
        worker_status['updated'] = False
        worker_status['complete'] = False
        worker_status['cycle'] = info['cycle']
    
    weights = global_model['weights']
    bias = global_model['bias']

    formated_parameters = OrderedDict([
        ('linear.weight', torch.tensor(weights,dtype=torch.float32)),
        ('linear.bias', torch.tensor(bias,dtype=torch.float32))
    ])

    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = workers_bucket,
        object_path = global_model_path,
        data = formated_parameters,
        metadata = {}
    )

    worker_status['stored'] = True
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = workers_bucket,
        object_path = worker_status_path,
        data = worker_status,
        metadata = {}
    )

    os.environ['STATUS'] = 'stored'

    return {'message': 'stored'}
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
    workers_bucket = 'workers'
    worker_experiment_folder = os.environ.get('WORKER_ID') + '/experiments'
    worker_status_path = worker_experiment_folder + '/status'
    central_status_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = workers_bucket,
        object_path = worker_status_path
    )
    worker_status = central_status_object['data']

    cycle_folder_path = worker_experiment_folder + '/' + str(worker_status['experiment']) + '/' + str(worker_status['cycle'])
    object_path = None
    object_data = None
    if type == 'metrics' or type == 'resources':
        if type == 'metrics':
            object_path = cycle_folder_path + '/metrics'
            for key,value in metrics.items():
                if key == 'name':
                    continue
                metric_name = prometheus_metrics['worker-local-names'][key]
                prometheus_metrics['worker-local'].labels(
                    date = datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
                    woid = worker_status['worker-id'],
                    neid = worker_status['network-id'],
                    cead = worker_status['central-address'],
                    woad = worker_status['worker-address'],
                    experiment = worker_status['experiment'], 
                    cycle = worker_status['cycle'],
                    metric = metric_name,
                ).set(value)
        if type == 'resources':
            object_path = cycle_folder_path + '/resources/' + area
            action_name = metrics['name']
            for key,value in metrics.items():
                if key == 'name':
                    continue
                metric_name = prometheus_metrics['worker-resources-names'][key]
                prometheus_metrics['worker-resources'].labels(
                    date = datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
                    woid = worker_status['worker-id'],
                    neid = worker_status['network-id'],
                    cead = worker_status['central-address'],
                    woad = worker_status['worker-address'],
                    experiment = worker_status['experiment'], 
                    cycle = worker_status['cycle'],
                    area = area,
                    name = action_name,
                    metric = metric_name,
                ).set(value)
            
        wanted_object = get_object_data_and_metadata(
            logger = logger,
            minio_client = minio_client,
            bucket_name = workers_bucket,
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
            bucket_name = workers_bucket,
            object_path = object_path,
            data = object_data,
            metadata = {}
        )
        #push_to_gateway('http:127.0.0.1:9091', job = 'central-', registry =  prometheus_registry) 
    
    return True