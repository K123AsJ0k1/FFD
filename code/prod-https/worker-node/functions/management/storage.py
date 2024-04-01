import torch  
import os
import time

from datetime import datetime
from collections import OrderedDict

from functions.platforms.minio import get_object_data_and_metadata, create_or_update_object, check_object
from functions.general import get_experiments_objects, set_experiments_objects
from functions.platforms.mlflow import start_experiment, check_experiment
# Refactored
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
    worker_status, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = ''
    )

    if worker_status is None:
        return {'message': 'no status'}

    if worker_status['complete']:
        return {'message': 'complete'}
    
    if not info['worker-id'] == worker_status['worker-id']:
        return {'message': 'incorrect'}

    #if not info['experiment-name'] == worker_status['experiment-name']:
    #    return {'message': 'incorrect'}
    
    #if not info['experiment'] == worker_status['experiment']:
    #    return {'message': 'incorrect'}
    
    if worker_status['stored'] and not worker_status['updated']:
        return {'message': 'ongoing'}
    
    os.environ['STATUS'] = 'storing'
    os.environ['EXP_NAME'] = str(info['experiment-name'])
    os.environ['EXP'] = str(info['experiment'])
    os.environ['CYCLE'] = str(info['cycle'])
    if info['model'] == None:
        worker_status['complete'] = True
        worker_status['cycle'] = info['cycle']
        
        experiment_times, _ = get_experiments_objects(
            file_lock = file_lock,
            logger = logger,
            minio_client = minio_client,
            object = 'experiment-times',
            replacer = ''
        )
        experiment_start = experiment_times['experiment-time-start']
        experiment_end = time.time()
        experiment_total = experiment_end - experiment_start
        experiment_times['experiment-time-end'] = experiment_end
        experiment_times['experiment-total-seconds'] = experiment_total
        set_experiments_objects(
            file_lock = file_lock,
            logger = logger,
            minio_client = minio_client,
            object = 'experiment-times',
            replacer = '',
            overwrite = True,
            object_data = experiment_times,
            object_metadata = {}
        )
    else:
        experiment_times = {
            'experiment-name': str(info['experiment-name']),
            'experiment': str(info['experiment']),
            'cycle': str(info['cycle']),
            'experiment-date': datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
            'experiment-time-start': time.time(),
            'experiment-time-end':0,
            'experiment-total-seconds': 0
        }
        
        set_experiments_objects(
            file_lock = file_lock,
            logger = logger,
            minio_client = minio_client,
            object = 'experiment-times',
            replacer = '',
            overwrite = False,
            object_data = experiment_times,
            object_metadata = {}
        )

        # change to enable different experiment names
        worker_experiment_name = 'worker-' + str(os.environ.get('WORKER_ID')) + '-' + str(info['experiment-name'])
        experiment_dict = check_experiment(
            logger = logger,
            mlflow_client = mlflow_client,
            experiment_name = worker_experiment_name
        )
        experiment_id = ''
        if experiment_dict is None:
            experiment_id = start_experiment(
                logger = logger,
                mlflow_client = mlflow_client,
                experiment_name = worker_experiment_name,
                experiment_tags = {}
            )
        else:
            experiment_id = experiment_dict.experiment_id
        worker_status['experiment-id'] = experiment_id
        
        set_experiments_objects(
            file_lock = file_lock,
            logger = logger,
            minio_client = minio_client,
            object = 'parameters',
            replacer = 'model',
            overwrite = False,
            object_data = info['model'],
            object_metadata = {}
        )

        set_experiments_objects(
            file_lock = file_lock,
            logger = logger,
            minio_client = minio_client,
            object = 'parameters',
            replacer = 'worker',
            overwrite = False,
            object_data = info['worker'],
            object_metadata = {}
        )

        worker_sample = df_data
        worker_sample_metadata = {
            'header': df_columns,
            'columns': len(df_columns),
            'rows': len(df_data)
        }

        set_experiments_objects(
            file_lock = file_lock,
            logger = logger,
            minio_client = minio_client,
            object = 'worker-sample',
            replacer = '',
            overwrite = False,
            object_data = worker_sample,
            object_metadata = worker_sample_metadata
        )
        '''
        worker_status['preprocessed'] = False
        worker_status['trained'] = False
        worker_status['updated'] = False
        worker_status['complete'] = False
        '''
        worker_status['experiment-name'] = info['experiment-name']
        worker_status['experiment'] = info['experiment']
        worker_status['cycle'] = info['cycle']
        
    weights = global_model['weights']
    bias = global_model['bias']

    formated_parameters = OrderedDict([
        ('linear.weight', torch.tensor(weights,dtype=torch.float32)),
        ('linear.bias', torch.tensor(bias,dtype=torch.float32))
    ])

    set_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'model',
        replacer = 'global-model',
        overwrite = False,
        object_data = formated_parameters,
        object_metadata = {}
    )

    worker_status['stored'] = True
    set_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = '',
        overwrite = True,
        object_data = worker_status,
        object_metadata = {}
    )
    
    os.environ['STATUS'] = 'stored'

    return {'message': 'stored'}
# Refactored
def store_metrics_resources_and_times( 
   file_lock: any,
   logger: any,
   minio_client: any,
   prometheus_registry: any,
   prometheus_metrics: any,
   type: str,
   area: str,
   metrics: any
) -> bool:
    worker_status, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = ''
    )    
    if not worker_status is None:
        object_name = ''
        replacer = ''
        if type == 'metrics' or type == 'resources' or type == 'times':
            if type == 'metrics':
                object_name = 'metrics'
                source = metrics['name']
                for key,value in metrics.items():
                    if key == 'name':
                        continue
                    metric_name = prometheus_metrics['local-name'][key]
                    prometheus_metrics['local'].labels(
                        date = datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
                        time = time.time(),
                        collector = 'worker-' + worker_status['worker-id'],
                        name = worker_status['experiment-name'],
                        experiment = worker_status['experiment'], 
                        cycle = worker_status['cycle'],
                        source = source,
                        metric = metric_name,
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
                        collector = 'worker-' + worker_status['worker-id'],
                        name = worker_status['experiment-name'], 
                        experiment = worker_status['experiment'],
                        cycle = worker_status['cycle'],
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
                        collector = 'worker-' + worker_status['worker-id'],
                        name = worker_status['experiment-name'],
                        experiment = worker_status['experiment'], 
                        cycle = worker_status['cycle'],
                        area = area,
                        source = source,
                        metric = metric_name
                    ).set(value)
            
            if not worker_status['experiment-name'] == '':
                wanted_data, _ = get_experiments_objects(
                    file_lock = file_lock,
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
                    file_lock = file_lock,
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