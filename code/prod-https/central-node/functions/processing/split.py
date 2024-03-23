import os
import numpy as np
import pandas as pd
import time
import psutil 

from functions.processing.data import data_augmented_sample
from functions.platforms.minio import get_object_data_and_metadata, create_or_update_object
from functions.general import format_metadata_dict, encode_metadata_lists_to_strings
from functions.management.storage import store_metrics_and_resources
# Refactored
def central_worker_data_split(
    file_lock: any,
    logger: any,
    minio_client: any
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

    status = store_metrics_and_resources(
        file_lock = file_lock,
        type = 'resources',
        subject = 'central',
        area = 'function',
        metrics = resource_metrics
    )

    return True
# Refactored and works
def split_data_between_workers(
    file_lock: any,
    logger: any,
    minio_client: any
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
    
    worker_status_path = status_folder_path + '/workers.txt'

    worker_status = get_file_data(
        file_lock = file_lock,
        file_path = worker_status_path
    )

    if worker_status is None:
        return False
    
    parameters_folder_path = 'parameters/experiment_' + str(current_experiment_number)
    central_parameters_path = parameters_folder_path + '/central.txt'
    worker_parameters_path = parameters_folder_path + '/worker.txt'

    central_parameters = get_file_data(
        file_lock = file_lock,
        file_path = central_parameters_path
    )

    if central_parameters is None:
        return False
    
    worker_parameters = get_file_data(
        file_lock = file_lock,
        file_path = worker_parameters_path
    )

    if worker_parameters is None:
        return False
    
    data_folder_path = 'data/experiment_' + str(current_experiment_number)
    worker_pool_path = data_folder_path + '/worker_pool.csv'
    worker_pool_df = get_file_data(
        file_lock = file_lock,
        file_path = worker_pool_path
    )

    os.environ['STATUS'] = 'worker splitting'
    
    available_workers = []
    for worker_key in worker_status.keys():
        worker_metadata = worker_status[worker_key]
        if not worker_metadata['stored'] and not worker_metadata['complete']: 
            available_workers.append(worker_key)
    
    # Could be reconsidered
    if not central_parameters['min-update-amount'] <= len(available_workers):
        return False
    # Might have concurrency issues
    # Format for worker data is worker_(id)_(cycle)_(size).csv
    if worker_parameters['data-augmentation']['active']:
        for worker_key in available_workers:
            worker_sample_df = data_augmented_sample(
                pool_df = worker_pool_df,
                sample_pool = worker_parameters['data-augmentation']['sample-pool'],
                ratio = worker_parameters['data-augmentation']['1-0-ratio']
            )
            sample_size = worker_sample_df.shape[0]
            data_path = data_folder_path + '/worker_' + worker_key + '_' + str(central_status['cycle']) + '_' + str(sample_size) + '.csv'
            store_file_data(
                file_lock = file_lock,
                replace = False,
                file_folder_path = data_folder_path,
                file_path = data_path,
                data = worker_sample_df
            )
    else:
        worker_df = worker_pool_df.sample(frac = 1)
        worker_dfs = np.array_split(worker_df, len(available_workers))
        index = 0
        for worker_key in available_workers:
            sample_size = worker_dfs[index].shape[0]
            data_path = data_folder_path + '/worker_' + worker_key + '_' + str(central_status['cycle']) + '_' + str(sample_size) + '.csv'
            store_file_data(
                file_lock = file_lock,
                replace = False,
                file_folder_path = data_folder_path,
                file_path = data_path,
                data = worker_dfs[index]
            )
            index = index + 1
    central_status['worker-split'] = True
    store_file_data(
        file_lock = file_lock,
        replace = True,
        file_folder_path = '',
        file_path = central_status_path,
        data = central_status
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

    status = store_metrics_and_resources(
        file_lock = file_lock,
        type = 'resources',
        subject = 'central',
        area = 'function',
        metrics = resource_metrics
    )

    return True  