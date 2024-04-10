import os
import time 

import numpy as np
import pandas as pd

from functions.management.objects import get_experiments_objects, set_experiments_objects
from functions.management.storage import store_metrics_resources_and_times
from functions.processing.data import data_augmented_sample
# Refactored and works
def central_worker_data_split(
    file_lock: any,
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any
) -> bool:
    time_start = time.time()

    central_status, _ = get_experiments_objects(
        file_lock = file_lock,
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
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'parameters',
        replacer = 'central'
    )

    worker_parameters, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'parameters',
        replacer = 'worker'
    )

    source_pool, details = get_experiments_objects(
        file_lock = file_lock,
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
        file_lock = file_lock,
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
        file_lock = file_lock,
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
        file_lock = file_lock,
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
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        type = 'times',
        area = 'function',
        metrics = resource_metrics
    )

    return True
# Refactored and works
def split_data_between_workers(
    file_lock: any,
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any
) -> bool:
    time_start = time.time()
    # There might be a concurrency problem with small resources in Oakestra
    central_status, _ = get_experiments_objects(
        file_lock = file_lock,
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
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'workers',
        replacer = ''
    )

    central_parameters, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'parameters',
        replacer = 'central'
    )

    worker_parameters, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'parameters',
        replacer = 'worker'
    )

    os.environ['STATUS'] = 'worker splitting'
    
    available_workers = []
    for worker_key in workers_status.keys():
        worker_status = workers_status[worker_key]
        if not worker_status['stored'] and not worker_status['complete']: 
            available_workers.append(worker_key)
    
    # Could be reconsidered
    if not central_parameters['min-update-amount'] <= len(available_workers):
        return False
    
    workers_data, workers_data_details = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'data',
        replacer = 'workers-pool' 
    )

    workers_pool_df = pd.DataFrame(workers_data, columns = workers_data_details['header'])

    if worker_parameters['data-augmentation']['active']:
        for worker_key in available_workers:
            worker_sample_df = data_augmented_sample(
                pool_df = workers_pool_df,
                sample_pool = worker_parameters['data-augmentation']['sample-pool'],
                ratio = worker_parameters['data-augmentation']['1-0-ratio']
            )
            
            object_data = worker_sample_df.values.tolist()
            object_metadata = {
                'header': worker_sample_df.columns.tolist(),
                'columns': str(len(worker_sample_df.columns.tolist())),
                'rows': str(worker_sample_df.shape[0])
            }
            set_experiments_objects(
                file_lock = file_lock,
                logger = logger,
                minio_client = minio_client,
                object = 'data-worker',
                replacer = worker_key,
                overwrite = True,
                object_data = object_data,
                object_metadata = object_metadata
            )
            
    else:
        worker_df = workers_pool_df.sample(frac = 1)
        worker_dfs = np.array_split(worker_df, len(available_workers))
        index = 0
        for worker_key in available_workers:
            worker_sample_df = worker_dfs[index]
             
            object_data = worker_sample_df.values.tolist()
            object_metadata = {
                'header': worker_sample_df.columns.tolist(),
                'columns': str(len(worker_sample_df.columns.tolist())),
                'rows': str(worker_sample_df.shape[0])
            }
            set_experiments_objects(
                file_lock = file_lock,
                logger = logger,
                minio_client = minio_client,
                object = 'data-worker',
                replacer = worker_key,
                overwrite = True,
                object_data = object_data,
                object_metadata = object_metadata
            )
            index = index + 1

    central_status['worker-split'] = True
    set_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = '',
        overwrite = True,
        object_data = central_status,
        object_metadata = {}
    )

    os.environ['STATUS'] = 'worker data split'
    logger.info('Worker data split')

    time_end = time.time()
    time_diff = (time_end - time_start) 
    resource_metrics = {
        'name': 'split-data-between-workers',
        'action-time-start': time_start,
        'action-time-end': time_end,
        'action-total-seconds': round(time_diff,5)
    }

    store_metrics_resources_and_times(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        type = 'times',
        area = 'function',
        metrics = resource_metrics
    )

    return True  
