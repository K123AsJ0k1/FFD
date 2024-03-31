import os
import numpy as np
import pandas as pd
import time
import psutil 

from functions.processing.data import data_augmented_sample
from functions.platforms.minio import get_object_data_and_metadata, create_or_update_object
from functions.general import get_experiments_objects, set_experiments_objects
from functions.management.storage import store_metrics_resources_and_times
# Refactored and works
def central_worker_data_split(
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any
) -> bool:
    time_start = time.time()

    central_status, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = ''
    )

    if central_status is None:
        return False
    
    if not central_status['start']:
        return False
    
    if central_status['complete']:
        return False
    
    if central_status['data-split']:
        return False
    
    os.environ['STATUS'] = 'splitting central and worker data'
    logger.info('Splitting data into central and workers pools')

    central_parameters, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'parameters',
        replacer = 'central'
    )

    worker_parameters, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'parameters',
        replacer = 'worker'
    )

    source_pool, details = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'data',
        replacer = 'source-pool'
    )
    
    source_df = pd.DataFrame(source_pool, columns = details['header'])
    
    splitted_data_df = source_df.drop('step', axis = 1)
    central_df = splitted_data_df.sample(n = central_parameters['sample-pool'])
    object_data = central_df.values.tolist()
    object_metadata = {
        'header': central_df.columns.tolist(),
        'columns': str(len(central_df.columns.tolist())),
        'rows': str(len(object_data))
    }
    set_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'data',
        replacer = 'central-pool',
        overwrite = True,
        object_data = object_data,
        object_metadata = object_metadata
    )

    central_indexes = central_df.index.tolist()
    splitted_data_df.drop(central_indexes)
    workers_df = splitted_data_df.sample(n = worker_parameters['sample-pool'])

    object_data = workers_df.values.tolist()
    object_metadata = {
        'header': workers_df.columns.tolist(),
        'columns': str(len(workers_df.columns.tolist())),
        'rows': str(len(object_data))
    }
    set_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'data',
        replacer = 'workers-pool',
        overwrite = True,
        object_data = object_data,
        object_metadata = object_metadata
    )

    central_status['data-split'] = True
    set_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = '',
        overwrite = True,
        object_data = central_status,
        object_metadata = {}
    )
    os.environ['STATUS'] = 'central and worker data split'
    logger.info('Central and workers pools created')
    
    time_end = time.time()
    time_diff = (time_end - time_start) 
    resource_metrics = {
        'name': 'central-worker-data-split',
        'action-time-start': time_start,
        'action-time-end': time_end,
        'action-total-seconds': round(time_diff,5)
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

    return True
# Refactor
def split_data_between_workers(
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any
) -> bool:
    time_start = time.time()

    central_status, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = ''
    )
    
    if central_status is None:
        return False
   
    if not central_status['start']:
        return False
    
    if central_status['complete']:
        return False
    
    if not central_status['preprocessed']:
        return False
    
    if central_status['worker-split']:
        return False
    
    os.environ['STATUS'] = 'splitting data between workers'
    logger.info('Splitting data between workers')

    workers_status, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'workers',
        replacer = ''
    )

    '''
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
    
    parameters_folder_path = experiment_folder_path + '/parameters'
    central_parameters_path = parameters_folder_path + '/central'
    worker_parameters_path = parameters_folder_path + '/worker'

    central_parameters_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_parameters_path
    )
    central_parameters = central_parameters_object['data']

    worker_parameters_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = worker_parameters_path
    )
    worker_parameters = worker_parameters_object['data']
    '''

    os.environ['STATUS'] = 'worker splitting'
    
    available_workers = []
    for worker_key in workers_status.keys():
        worker_status = workers_status[worker_key]
        if not worker_status['stored'] and not worker_status['complete']: 
            available_workers.append(worker_key)
    
    # Could be reconsidered
    if not central_parameters['min-update-amount'] <= len(available_workers):
        return False
    
    data_folder_path = experiment_folder_path + '/data'
    worker_pool_path = data_folder_path + '/worker-pool'
    worker_pool_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = worker_pool_path
    )
    data_columns = format_metadata_dict(worker_pool_object['metadata'])['header']
    worker_pool_df = pd.DataFrame(worker_pool_object['data'], columns = data_columns)

    if worker_parameters['data-augmentation']['active']:
        for worker_key in available_workers:
            worker_sample_df = data_augmented_sample(
                pool_df = worker_pool_df,
                sample_pool = worker_parameters['data-augmentation']['sample-pool'],
                ratio = worker_parameters['data-augmentation']['1-0-ratio']
            )
            worker_sample_path = cycle_folder_path + '/data/' + worker_key 
            object_data = worker_sample_df.values.tolist()
            object_metadata = encode_metadata_lists_to_strings({
                'header': worker_sample_df.columns.tolist(),
                'columns': str(len(worker_sample_df.columns.tolist())),
                'rows': str(worker_sample_df.shape[0])
            })
            create_or_update_object(
                logger = logger,
                minio_client = minio_client,
                bucket_name = central_bucket,
                object_path = worker_sample_path,
                data = object_data,
                metadata = object_metadata
            )
    else:
        worker_df = worker_pool_df.sample(frac = 1)
        worker_dfs = np.array_split(worker_df, len(available_workers))
        index = 0
        for worker_key in available_workers:
            worker_sample_df = worker_dfs[index]
            worker_sample_path = cycle_folder_path + '/data/' + worker_key 
            object_data = worker_sample_df.values.tolist()
            object_metadata = encode_metadata_lists_to_strings({
                'header': worker_sample_df.columns.tolist(),
                'columns': str(len(worker_sample_df.columns.tolist())),
                'rows': str(worker_sample_df.shape[0])
            })
            create_or_update_object(
                logger = logger,
                minio_client = minio_client,
                bucket_name = central_bucket,
                object_path = worker_sample_path,
                data = object_data,
                metadata = object_metadata
            )
            index = index + 1

    central_status['worker-split'] = True
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path,
        data = central_status,
        metadata = {}
    )

    os.environ['STATUS'] = 'worker data split'
    logger.info('Worker data split')

    time_end = time.time()
    cpu_end = this_process.cpu_percent(interval=0.2)
    mem_end = psutil.virtual_memory().used 
    disk_end = psutil.disk_usage('.').used

    time_diff = (time_end - time_start) 
    cpu_diff = cpu_end - cpu_start 
    mem_diff = (mem_end - mem_start) 
    disk_diff = (disk_end - disk_start)

    resource_metrics = {
        'name': 'split-data-between-workers',
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
