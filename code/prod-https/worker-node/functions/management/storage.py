import pandas as pd
import torch  
import os
import json
import psutil
from pathlib import Path
from datetime import datetime
import time

from collections import OrderedDict

from functions.platforms.minio import get_object_data_and_metadata, create_or_update_object
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

    worker_resources_path = 'resources/experiment_' + str(current_experiment_number) + '/worker.txt'
    worker_resources = get_file_data(
        file_lock = file_lock,
        file_path = worker_resources_path
    )
    
    if worker_status['complete']:
        experiment_start = worker_resources['general']['times']['experiment-time-start']
        experiment_end = time.time()
        experiment_total = experiment_end - experiment_start
        worker_resources['general']['times']['experiment-time-end'] = experiment_end
        worker_resources['general']['times']['experiment-total-seconds'] = experiment_total
    else:
        if str(worker_status['cycle']) == '1':
            worker_resources['general']['times']['experiment-date'] = datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f')
            worker_resources['general']['times']['experiment-time-start'] = time.time()

    store_file_data(
        file_lock = file_lock,
        replace = True,
        file_folder_path = '',
        file_path = worker_resources_path,
        data = worker_resources
    )

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
'''
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
                return False

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
'''