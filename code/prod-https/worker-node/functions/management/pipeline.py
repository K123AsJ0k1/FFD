import time

from functions.monitoring import get_server_resource_usage, get_system_resource_usage

from functions.management.objects import get_experiments_objects, set_experiments_objects
from functions.management.storage import store_metrics_resources_and_times

from functions.processing.data import preprocess_into_train_test_and_eval_tensors

from functions.management.update import send_info_to_central, send_update_to_central
from functions.training.model import local_model_training

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
def status_pipeline(
    task_file_lock: any,
    task_logger: any,
    task_minio_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any
):
    time_start = time.time()

    # Works
    status = send_info_to_central(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Status sending:' + str(status))

    time_end = time.time()
    time_diff = (time_end - time_start) 
    action_time = {
        'name': 'status-pipeline',
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
# Refactoroed and works
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
    status = preprocess_into_train_test_and_eval_tensors(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )

    if status:
        worker_status, _ = get_experiments_objects(
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

        experiment_times[str(worker_status['cycle'])] = {
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

    task_logger.info('Data preprocessing:' + str(status))

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
    task_prometheus_metrics: any
): 
    time_start = time.time()

    # Works
    status = local_model_training(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        mlflow_client = task_mlflow_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Model training:' + str(status))

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
# Refactored and works
def update_pipeline(
    task_file_lock: any,
    task_logger: any,
    task_minio_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any
):
    time_start = time.time()

    # Works
    status = send_update_to_central(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Update sending:' + str(status))

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
