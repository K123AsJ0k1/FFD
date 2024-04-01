import os
import numpy as np
import pandas as pd
import torch 
import time
import psutil

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from functions.general import get_experiments_objects, set_experiments_objects
from functions.management.storage import store_metrics_resources_and_times

# Created and works
def data_augmented_sample(
    pool_df: any,
    sample_pool: int,
    ratio: int
) -> any:
    fraud_cases = pool_df[pool_df['isFraud'] == 1]
    non_fraud_cases = pool_df[pool_df['isFraud'] == 0]

    wanted_fraud_amount = int(sample_pool * ratio)
    wanted_non_fraud_amount = sample_pool-wanted_fraud_amount

    frauds_df = fraud_cases.sample(n = wanted_fraud_amount, replace = True)
    non_fraud_df = non_fraud_cases.sample(n = wanted_non_fraud_amount, replace = True)

    augmented_sample_df = pd.concat([frauds_df,non_fraud_df])
    randomized_sample_df = augmented_sample_df.sample(frac = 1, replace = False)
    return randomized_sample_df
# Refactored and works
def preprocess_into_train_test_and_evaluate_tensors(
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

    if not central_status['data-split']:
        return False

    if central_status['preprocessed']:
        return False
    
    os.environ['STATUS'] = 'preprocessing into tensors'
    logger.info('Preprocessing into tensors')
    
    central_parameters, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'parameters',
        replacer = 'central'
    )
    
    model_parameters, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'parameters',
        replacer = 'model'
    )
    
    central_pool, details = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'data',
        replacer = 'central-pool'
    )
    
    central_df = pd.DataFrame(central_pool, columns = details['header'])
    preprocessed_df = None
    if central_parameters['data-augmentation']['active']:
        used_data_df = data_augmented_sample(
            pool_df = central_df,
            sample_pool = central_parameters['data-augmentation']['sample-pool'],
            ratio = central_parameters['data-augmentation']['1-0-ratio']
        )
        preprocessed_df = used_data_df[model_parameters['used-columns']]
    else:
        preprocessed_df = central_df[model_parameters['used-columns']]
    
    for column in model_parameters['scaled-columns']:
        mean = preprocessed_df[column].mean()
        std_dev = preprocessed_df[column].std()
        preprocessed_df[column] = (preprocessed_df[column] - mean)/std_dev

    X = preprocessed_df.drop(model_parameters['target-column'], axis = 1).values
    y = preprocessed_df[model_parameters['target-column']].values
        
    X_eval, X_train_test, y_eval, y_train_test = train_test_split(
        X, 
        y, 
        train_size = central_parameters['eval-ratio'], 
        random_state = model_parameters['seed']
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_test, 
        y_train_test, 
        train_size = central_parameters['train-ratio'], 
        random_state = model_parameters['seed']
    )

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    X_eval = np.array(X_eval, dtype=np.float32)

    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)
    y_eval = np.array(y_eval, dtype=np.int32)
    
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
    
    central_status['preprocessed'] = True
    central_status['train-amount'] = X_train.shape[0]
    central_status['test-amount'] = X_test.shape[0]
    central_status['eval-amount'] = X_eval.shape[0]
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