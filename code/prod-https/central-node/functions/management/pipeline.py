import time
import os

from datetime import datetime

from functions.monitoring import get_system_resource_usage, get_server_resource_usage

from functions.management.update import send_context_to_workers
from functions.management.objects import get_experiments_objects, set_experiments_objects
from functions.management.storage import store_metrics_resources_and_times

from functions.platforms.mlflow import start_experiment, check_experiment

from functions.processing.split import central_worker_data_split, split_data_between_workers
from functions.processing.data import preprocess_into_train_test_and_evaluate_tensors

from functions.training.model import initial_model_training
from functions.training.aggregation import update_global_model, evalute_global_model

# Refactored and works
def start_pipeline( 
    file_lock: any,
    logger: any,
    minio_client: any,
    mlflow_client: any,
    prometheus_registry: any,
    prometheus_metrics: any,
    experiment: any,
    parameters: any,
    df_data: list,
    df_columns: list
):  
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
    
    if central_status['start'] and not central_status['complete']:
        return False
    
    if central_status['start'] and central_status['complete']:
        central_status['data-split'] = False
        central_status['preprocessed'] = False
        central_status['trained'] = False
        central_status['worker-split'] = False
        central_status['sent'] = False
        central_status['updated'] = False
        central_status['evaluated'] = False
        central_status['complete'] = False
        central_status['experiment'] = central_status['experiment'] + 1
        central_status['cycle'] = 1

    central_experiment_name = 'central-' + experiment['name']
    experiment_dict = check_experiment(
        logger = logger,
        mlflow_client = mlflow_client,
        experiment_name = central_experiment_name
    )    
    os.environ['EXP_NAME'] = central_experiment_name
    os.environ['EXP'] = str(central_status['experiment'])
    os.environ['CYCLE'] = str(central_status['cycle'])
    central_status['experiment-name'] = central_experiment_name
    experiment_id = ''
    if experiment_dict is None:
        experiment_id = start_experiment(
            logger = logger,
            mlflow_client = mlflow_client,
            experiment_name = central_experiment_name,
            experiment_tags = experiment['tags']
        )
    else:
        experiment_id = experiment_dict.experiment_id
    central_status['experiment-id'] = experiment_id
    
    times = {
        'experiment-date': datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
        'experiment-time-start': time.time(),
        'experiment-time-end':0,
        'experiment-total-seconds': 0,
    }

    set_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'experiment-times',
        replacer = '',
        overwrite = True,
        object_data = times,
        object_metadata = {}
    )

    for name in parameters.keys():
        template_name = name + '-template'
        modified_parameters, _ = get_experiments_objects(
            file_lock = file_lock,
            logger = logger,
            minio_client = minio_client,
            object = template_name,
            replacer = ''
        )
        given_parameters = parameters[name]
        for key in modified_parameters.keys():
            modified_parameters[key] = given_parameters[key]
        set_experiments_objects(
            file_lock = file_lock,
            logger = logger,
            minio_client = minio_client,
            object = 'parameters',
            replacer = name,
            overwrite = True,
            object_data = modified_parameters,
            object_metadata = {}
        )
        
    metadata = {
        'header': df_columns, 
        'columns': len(df_columns), 
        'rows': str(len(df_data))
    }
    set_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'data',
        replacer = 'source-pool',
        overwrite = True,
        object_data = df_data,
        object_metadata = metadata
    )

    central_status['start'] = True
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

    time_end = time.time()
    time_diff = (time_end - time_start) 
    action_time = {
        'name': 'start-pipeline',
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
        metrics = action_time
    )

    return True
# Created and works
def system_monitoring(
    task_file_lock: any,
    task_logger: any,
    task_minio_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any
):
    time_start = time.time()

    system_resources = get_system_resource_usage()
    store_metrics_resources_and_times(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics,
        type = 'resources',
        area = '',
        metrics = system_resources
    )

    time_end = time.time()
    time_diff = (time_end - time_start) 
    action_time = {
        'name': 'system-monitoring',
        'action-time-start': time_start,
        'action-time-end': time_end,
        'action-total-seconds': round(time_diff,5)
    }

    store_metrics_resources_and_times(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics,
        type = 'times',
        area = 'task',
        metrics = action_time
    )
# Created and works
def server_monitoring(
    task_file_lock: any,
    task_logger: any,
    task_minio_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any
):
    time_start = time.time()

    server_resources = get_server_resource_usage()
    store_metrics_resources_and_times(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics,
        type = 'resources',
        area = '',
        metrics = server_resources
    )

    time_end = time.time()
    time_diff = (time_end - time_start) 
    action_time = {
        'name': 'server-monitoring',
        'action-time-start': time_start,
        'action-time-end': time_end,
        'action-total-seconds': round(time_diff,5)
    }

    store_metrics_resources_and_times(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics,
        type = 'times',
        area = 'task',
        metrics = action_time
    )
# Refactored and works
def data_pipeline(
    task_file_lock: any,
    task_logger: any,
    task_minio_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any
):
    time_start = time.time()

    cycle_start = time.time()
    # Works
    status = central_worker_data_split(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Central-worker data split:' + str(status))

    if status:
        central_status, _ = get_experiments_objects(
            file_lock = task_file_lock,
            logger = task_logger,
            minio_client = task_minio_client,
            object = 'status',
            replacer = ''
        )

        experiment_times, _ = get_experiments_objects(
            file_lock = task_file_lock,
            logger = task_logger,
            minio_client = task_minio_client,
            object = 'experiment-times',
            replacer = ''
        )
        
        experiment_times[str(central_status['cycle'])] = {
            'cycle-time-start':cycle_start,
            'cycle-time-end': 0,
            'cycle-total-seconds': 0
        }

        set_experiments_objects(
            file_lock = task_file_lock,
            logger = task_logger,
            minio_client = task_minio_client,
            object = 'experiment-times',
            replacer = '',
            overwrite = True,
            object_data = experiment_times,
            object_metadata = {}
        )
    # Works
    status = preprocess_into_train_test_and_evaluate_tensors(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Central pool preprocessing:' + str(status))
    
    time_end = time.time()
    time_diff = (time_end - time_start) 
    action_time = {
        'name': 'data-pipeline',
        'action-time-start': time_start,
        'action-time-end': time_end,
        'action-total-seconds': round(time_diff,5)
    }

    store_metrics_resources_and_times(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics,
        type = 'times',
        area = 'task',
        metrics = action_time
    )

# Refactored and works
def model_pipeline(
    task_file_lock: any,
    task_logger: any,
    task_minio_client: any,
    task_mlflow_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any,
):  
    time_start = time.time()

    # Works
    status = initial_model_training(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        mlflow_client = task_mlflow_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Initial model training:' + str(status))

    time_end = time.time()
    time_diff = (time_end - time_start) 
    action_time = {
        'name': 'model-pipeline',
        'action-time-start': time_start,
        'action-time-end': time_end,
        'action-total-seconds': round(time_diff,5)
    }

    store_metrics_resources_and_times(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics,
        type = 'times',
        area = 'task',
        metrics = action_time
    )
# Refactor and works
def update_pipeline(
    task_file_lock: any,
    task_logger: any,
    task_minio_client: any,
    task_mlflow_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any,
):
    time_start = time.time()

    # Works
    status = split_data_between_workers(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Worker data split:' + str(status))
    # Works
    status = send_context_to_workers(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        mlflow_client = task_mlflow_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Worker context sending:' + str(status))

    time_end = time.time()
    time_diff = (time_end - time_start) 
    action_time = {
        'name': 'update-pipeline',
        'action-time-start': time_start,
        'action-time-end': time_end,
        'action-total-seconds': round(time_diff,5)
    }

    store_metrics_resources_and_times(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics,
        type = 'times',
        area = 'task',
        metrics = action_time
    )
# Refactor and works
def aggregation_pipeline(
    task_file_lock: any,
    task_logger: any,
    task_minio_client: any,
    task_mlflow_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any,
):
    time_start = time.time()

    # Works
    status = update_global_model(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Updating global model:' + str(status))
    # Works
    status = evalute_global_model(
        file_lock = task_file_lock,
        logger = task_logger,
        mlflow_client = task_mlflow_client,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Global model evaluation:' + str(status))

    time_end = time.time()
    time_diff = (time_end - time_start) 
    action_time = {
        'name': 'aggregation-pipeline',
        'action-time-start': time_start,
        'action-time-end': time_end,
        'action-total-seconds': round(time_diff,5)
    }

    store_metrics_resources_and_times(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics,
        type = 'times',
        area = 'task',
        metrics = action_time
    )
