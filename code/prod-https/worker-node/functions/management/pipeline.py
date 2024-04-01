from functions.training.update import send_info_to_central, send_update_to_central
from functions.processing.data import preprocess_into_train_test_and_eval_tensors
from functions.training.model import local_model_training
from functions.general import get_experiments_objects, set_experiments_objects, get_system_resource_usage, get_server_resource_usage
from functions.management.storage import store_metrics_resources_and_times

import time
# Created and works
def system_monitoring(
    task_logger: any,
    task_minio_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any
):
    system_resources = get_system_resource_usage()
    store_metrics_resources_and_times(
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics,
        type = 'resources',
        area = '',
        metrics = system_resources
    )
# Created and works
def server_monitoring(
    task_logger: any,
    task_minio_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any
):
    server_resources = get_server_resource_usage()
    store_metrics_resources_and_times(
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics,
        type = 'resources',
        area = '',
        metrics = server_resources
    )
# Refactored
def status_pipeline(
    task_logger: any,
    task_minio_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any
):
    # Check
    status = send_info_to_central(
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Status sending:' + str(status))
# Refactoroed
def data_pipeline(
    task_logger: any,
    task_minio_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any
):
    cycle_start = time.time()
    # Check
    status = preprocess_into_train_test_and_eval_tensors(
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )

    if status:
        worker_status, _ = get_experiments_objects(
            logger = task_logger,
            minio_client = task_minio_client,
            object = 'status',
            replacer = ''
        )

        experiment_times, _ = get_experiments_objects(
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
            logger = task_logger,
            minio_client = task_minio_client,
            object = 'experiment-times',
            replacer = '',
            overwrite = True,
            object_data = experiment_times,
            object_metadata = {}
        )

    task_logger.info('Data preprocessing:' + str(status))
# Refactored
def model_pipeline(
    task_logger: any,
    task_minio_client: any,
    task_mlflow_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any
): 
    # Check
    status = local_model_training(
        logger = task_logger,
        minio_client = task_minio_client,
        mlflow_client = task_mlflow_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Model training:' + str(status))
# Refactored
def update_pipeline(
    task_logger: any,
    task_minio_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any
):
    # Check
    status = send_update_to_central(
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Update sending:' + str(status))
