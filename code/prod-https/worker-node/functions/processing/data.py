import numpy as np
import torch  
import os
import time 
import psutil
import pandas as pd
 
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from functions.management.storage import store_metrics_and_resources
from functions.platforms.minio import get_object_data_and_metadata, create_or_update_object
from functions.general import format_metadata_dict
# Refactored and works
def preprocess_into_train_test_and_eval_tensors(
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
        return False
    
    if worker_status['complete']:
        return False

    if not worker_status['stored']:
        return False

    if worker_status['preprocessed']:
        return False

    os.environ['STATUS'] = 'preprocessing into tensors'
    logger.info('Preprocessing into tensors')
    experiment_folder_path = worker_experiments_folder + '/' + str(worker_status['experiment'])
    cycle_folder_path = experiment_folder_path + '/' + str(worker_status['cycle'])
    
    parameters_folder_path = experiment_folder_path + '/parameters'
    model_parameters_path = parameters_folder_path + '/model'
    model_parameters_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = workers_bucket,
        object_path = model_parameters_path
    )
    model_parameters = model_parameters_object['data']

    worker_parameters_path = parameters_folder_path + '/worker'
    worker_parameters_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = workers_bucket,
        object_path = worker_parameters_path
    )
    worker_parameters = worker_parameters_object['data']
    sample_df_path = cycle_folder_path + '/worker-sample'
    sample_df_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = workers_bucket,
        object_path = sample_df_path
    )

    data_columns = format_metadata_dict(sample_df_object['metadata'])['header']
    source_df = pd.DataFrame(sample_df_object['data'], columns = data_columns)
   
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

    tensor_folder_path = cycle_folder_path + '/tensors'
    
    train_tensor_path = tensor_folder_path + '/train'
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = workers_bucket,
        object_path = train_tensor_path,
        data = train_tensor,
        metadata = {}
    )
    
    test_tensor_path = tensor_folder_path + '/test'
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = workers_bucket,
        object_path = test_tensor_path,
        data = test_tensor,
        metadata = {}
    )

    eval_tensor_path = tensor_folder_path + '/eval' 
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = workers_bucket,
        object_path = eval_tensor_path,
        data = eval_tensor,
        metadata = {}
    )
    
    worker_status['preprocessed'] = True
    worker_status['train-amount'] = X_train.shape[0]
    worker_status['test-amount'] = X_test.shape[0]
    worker_status['eval-amount'] = X_eval.shape[0]
    
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = workers_bucket,
        object_path = worker_status_path,
        data = worker_status,
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