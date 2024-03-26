import os
import numpy as np
import pandas as pd
import time
import psutil 

from functions.processing.data import data_augmented_sample
from functions.platforms.minio import get_object_data_and_metadata, create_or_update_object
from functions.general import format_metadata_dict, encode_metadata_lists_to_strings
from functions.management.storage import store_metrics_and_resources
# Refactored and works 
def central_worker_data_split(
    file_lock: any,
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any
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
    
    if not central_status['start']:
        return False
    
    if central_status['complete']:
        return False
    
    if central_status['data-split']:
        return False
    
    os.environ['STATUS'] = 'splitting central and worker data'
    logger.info('Splitting data into central and workers pools')

    experiment_folder = experiments_folder + '/' + str(central_status['experiment'])
    
    parameter_folder_path = experiment_folder + '/parameters'
    
    central_parameters_path = parameter_folder_path + '/central'
    central_parameters_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_parameters_path
    )
    central_parameters = central_parameters_object['data']

    if central_parameters is None:
        return False

    worker_parameters_path = parameter_folder_path + '/worker'
    worker_parameters_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = worker_parameters_path
    )
    worker_parameters = worker_parameters_object['data']

    if worker_parameters is None:
        return False
    
    data_folder_path = experiment_folder + '/data'
    source_data_path = data_folder_path + '/source'

    source_data_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = source_data_path
    )
    data_columns = format_metadata_dict(source_data_object['metadata'])['columns']
    source_df = pd.DataFrame(source_data_object['data'], columns = data_columns)
    splitted_data_df = source_df.drop('step', axis = 1)
    
    central_data_pool = splitted_data_df.sample(n = central_parameters['sample-pool'])
    central_pool_path = data_folder_path + '/central-pool'
    
    object_data = central_data_pool.values.tolist()
    object_metadata = encode_metadata_lists_to_strings({'columns': central_data_pool.columns.tolist()})
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_pool_path,
        data = object_data,
        metadata = object_metadata
    )

    central_indexes = central_data_pool.index.tolist()
    splitted_data_df.drop(central_indexes)
    worker_data_pool = splitted_data_df.sample(n = worker_parameters['sample-pool'])
    worker_pool_path = data_folder_path + '/worker-pool'

    object_data = worker_data_pool.values.tolist()
    object_metadata = encode_metadata_lists_to_strings({'columns': worker_data_pool.columns.tolist()})
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = worker_pool_path,
        data = object_data,
        metadata = object_metadata
    )

    central_status['data-split'] = True
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path,
        data = central_status,
        metadata = {}
    )
    os.environ['STATUS'] = 'central and worker data split'
    logger.info('Central and workers pools created')
    
    time_end = time.time()
    cpu_end = this_process.cpu_percent(interval=0.2)
    mem_end = psutil.virtual_memory().used 
    disk_end = psutil.disk_usage('.').used

    time_diff = (time_end - time_start) 
    cpu_diff = cpu_end - cpu_start 
    mem_diff = (mem_end - mem_start) 
    disk_diff = (disk_end - disk_start)  

    resource_metrics = {
        'name': 'central-worker-data-split',
        'time-seconds': round(time_diff,5),
        'cpu-percentage': round(cpu_diff,5),
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
# Refactored
def split_data_between_workers(
    file_lock: any,
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any
) -> bool:
    # For some reason this gets falses
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
    worker_pool_path = data_folder_path + '/worker_pool'
    worker_pool_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = worker_pool_path
    )
    data_columns = format_metadata_dict(worker_pool_object['metadata'])['columns']
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
                'columns': worker_sample_df.columns.tolist(),
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
                'columns': worker_sample_df.columns.tolist(),
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
