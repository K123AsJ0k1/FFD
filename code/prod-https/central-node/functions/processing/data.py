import os
import numpy as np
import pandas as pd
import torch 
import time
import psutil

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from functions.platforms.minio import get_object_data_and_metadata, create_or_update_object
from functions.general import format_metadata_dict
from functions.management.storage import store_metrics_and_resources

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

    if not central_status['data-split']:
        return False

    if central_status['preprocessed']:
        return False
    
    os.environ['STATUS'] = 'preprocessing into tensors'
    logger.info('Preprocessing into tensors')

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

    model_parameters_path = parameter_folder_path + '/model' 
    model_parameters_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = model_parameters_path
    )
    model_parameters = model_parameters_object['data']
    data_folder = experiment_folder + '/data'

    central_pool_path = data_folder + '/central-pool'
    central_pool_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_pool_path
    )
    data_columns = format_metadata_dict(central_pool_object['metadata'])['header']
    central_data_df = pd.DataFrame(central_pool_object['data'], columns = data_columns)

    preprocessed_df = None
    if central_parameters['data-augmentation']['active']:
        used_data_df = data_augmented_sample(
            pool_df = central_data_df,
            sample_pool = central_parameters['data-augmentation']['sample-pool'],
            ratio = central_parameters['data-augmentation']['1-0-ratio']
        )
        preprocessed_df = used_data_df[model_parameters['used-columns']]
    else:
        preprocessed_df = central_data_df[model_parameters['used-columns']]
    
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
    # tensors have format train/test/eval_(cycle)
    tensors_folder_path = experiment_folder + '/tensors'
    train_tensor_path = tensors_folder_path  + '/train'
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = train_tensor_path ,
        data = train_tensor,
        metadata = {}
    )
 
    test_tensor_path = tensors_folder_path  + '/test'
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = test_tensor_path ,
        data = test_tensor,
        metadata = {}
    )
    eval_tensor_path = tensors_folder_path  + '/eval'
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = eval_tensor_path ,
        data = eval_tensor,
        metadata = {}
    )

    central_status['preprocessed'] = True
    central_status['train-amount'] = X_train.shape[0]
    central_status['test-amount'] = X_test.shape[0]
    central_status['eval-amount'] = X_eval.shape[0]
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path,
        data = central_status,
        metadata = {}
    )
    
    os.environ['STATUS'] = 'tensors created'
    logger.info('Tensors created')

    time_end = time.time()
    cpu_end = this_process.cpu_percent(interval=0.2)
    mem_end = psutil.virtual_memory().used 
    disk_end = psutil.disk_usage('.').used
    
    time_diff = (time_end - time_start) 
    cpu_diff = cpu_end - cpu_start 
    mem_diff = (mem_end - mem_start) 
    disk_diff = (disk_end - disk_start)

    resource_metrics = {
        'name': 'preprocess-into-train-test-and-evalute-tensors',
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