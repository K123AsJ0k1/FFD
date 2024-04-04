
import os
import time 

import torch  
import numpy as np
import pandas as pd
 
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from functions.management.objects import get_experiments_objects, set_experiments_objects

from functions.management.storage import store_metrics_resources_and_times

# Refactored and works
def preprocess_into_train_test_and_eval_tensors(
    file_lock: any,
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any
) -> bool:
    time_start = time.time()

    worker_status, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = ''
    )

    if worker_status is None:
        return False
    
    if worker_status['complete']:
        return False

    if not worker_status['stored']:
        return False

    if worker_status['preprocessed']:
        return False

    os.environ['STATUS'] = 'preprocessing into tensors'
    logger.info('Preprocessing into tensors')

    model_parameters, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'parameters',
        replacer = 'model'
    )

    worker_parameters, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'parameters',
        replacer = 'worker'
    )

    worker_sample, worker_sample_details = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'worker-sample',
        replacer = ''
    )

    source_df = pd.DataFrame(worker_sample, columns = worker_sample_details['header'])
   
    preprocessed_df = source_df[model_parameters['used-columns']]
    for column in model_parameters['scaled-columns']:
        mean = preprocessed_df[column].mean()
        std_dev = preprocessed_df[column].std()
        preprocessed_df[column] = (preprocessed_df[column] - mean)/std_dev

    X = preprocessed_df.drop(model_parameters['target-column'], axis = 1).values
    y = preprocessed_df[model_parameters['target-column']].values
        
    X_eval, X_train_test, y_eval, y_train_test = train_test_split(
        X, 
        y, 
        train_size = worker_parameters['eval-ratio'], 
        random_state = model_parameters['seed']
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_test, 
        y_train_test, 
        train_size = worker_parameters['train-ratio'], 
        random_state = model_parameters['seed']
    )

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    X_eval = np.array(X_eval, dtype=np.float32)

    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)
    y_eval = np.array(y_eval, dtype=np.float32)
    
    train_tensor = TensorDataset(
        torch.tensor(X_train), 
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_tensor = TensorDataset(
        torch.tensor(X_test), 
        torch.tensor(y_test, dtype=torch.float32)
    )
    eval_tensor = TensorDataset(
        torch.tensor(X_eval), 
        torch.tensor(y_eval, dtype=torch.float32)
    )

    set_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'tensors',
        replacer = 'train',
        overwrite = True,
        object_data = train_tensor,
        object_metadata = {}
    )

    set_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'tensors',
        replacer = 'test',
        overwrite = True,
        object_data = test_tensor,
        object_metadata = {}
    )

    set_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'tensors',
        replacer = 'eval',
        overwrite = True,
        object_data = eval_tensor,
        object_metadata = {}
    )

    worker_status['preprocessed'] = True
    worker_status['train-amount'] = X_train.shape[0]
    worker_status['test-amount'] = X_test.shape[0]
    worker_status['eval-amount'] = X_eval.shape[0]
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
    
    os.environ['STATUS'] = 'tensors created'
    logger.info('Tensors created')

    time_end = time.time()
    time_diff = (time_end - time_start) 

    resource_metrics = {
        'name': 'preprocess-into-train-test-and-evalute-tensors',
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