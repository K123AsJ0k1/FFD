from functions.platforms.minio import get_object_data_and_metadata, create_or_update_object
from functions.general import encode_metadata_lists_to_strings
from functions.processing.split import central_worker_data_split, split_data_between_workers
from functions.processing.data import preprocess_into_train_test_and_evaluate_tensors
from functions.training.model import initial_model_training
from functions.platforms.mlflow import start_experiment
from functions.management.update import send_context_to_workers
from datetime import datetime
import time
# Refactored and works
def start_pipeline(
    file_lock: any,
    logger: any,
    mlflow_client: any,
    minio_client: any,
    experiment: any,
    parameters: any,
    df_data: list,
    df_columns: list
):  
    central_bucket = 'central'
    central_status_path = 'experiments/status'
    central_status_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path
    )
    central_status = central_status_object['data']

    if central_status is None:
        return False
    
    if central_status['start'] and not central_status['complete']:
        return False
    
    experiment_id = start_experiment(
        logger = logger,
        mlflow_client = mlflow_client,
        experiment_name = experiment['name'],
        experiment_tags = experiment['tags']
    )
    
    times = {
        'experiment-date': datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
        'experiment-time-start': time.time(),
        'experiment-time-end':0,
        'experiment-total-seconds': 0,
    }
    object_path = 'experiments/' + str(central_status['experiment']) + '/times'
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = object_path,
        data = times,
        metadata = {}
    )
    
    for name in parameters.keys():
        template_parameter_path = 'experiments/templates' + '/' + str(name) + '-parameters'
        template_parameter_object = get_object_data_and_metadata(
            logger = logger,
            minio_client = minio_client,
            bucket_name = central_bucket ,
            object_path = template_parameter_path
        )
        template_parameters = template_parameter_object['data']
        given_parameters = parameters[name]
        for key in template_parameters.keys():
            template_parameters[key] = given_parameters[key]
        
        modified_parameter_path = 'experiments/' + str(central_status['experiment']) + '/parameters/' + str(name)
        create_or_update_object(
            logger = logger,
            minio_client = minio_client,
            bucket_name = central_bucket,
            object_path = modified_parameter_path,
            data = template_parameters,
            metadata = {}
        )
    formatted_metadata = encode_metadata_lists_to_strings({'columns': df_columns})
    source_data_path = 'experiments/' + str(central_status['experiment']) + '/data/source'
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = source_data_path,
        data = df_data,
        metadata = formatted_metadata
    )
    central_status['experiment-id'] = experiment_id
    central_status['start'] = True
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path,
        data = central_status,
        metadata = {}
    )
    
    return True
# Refactored and works
def processing_pipeline(
    task_file_lock: any,
    task_logger: any,
    task_minio_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any
):
    # Works
    status = central_worker_data_split(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Central-worker data split:' + str(status))
    # Works
    status = preprocess_into_train_test_and_evaluate_tensors(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Central pool preprocessing:' + str(status))
# Refactored and works
def model_pipeline(
    task_file_lock: any,
    task_logger: any,
    task_minio_client: any,
    task_mlflow_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any,
):  
    # Works
    status = initial_model_training(
        file_lock = task_file_lock,
        logger = task_logger,
        mlflow_client = task_mlflow_client,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Initial model training:' + str(status))
# Refactor
def update_pipeline(
    task_file_lock: any,
    task_logger: any
):
    # Check
    status = split_data_between_workers(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Worker data split:' + str(status))
    # Check
    status = send_context_to_workers(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Worker context sending:' + str(status))
'''
# Refactor
def aggregation_pipeline(
    task_file_lock: any,
    task_logger: any
):
    # 
    status = update_global_model(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Updating global model:' + str(status))
    # 
    status = evalute_global_model(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Global model evaluation:' + str(status))
'''
