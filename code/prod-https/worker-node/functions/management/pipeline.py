from functions.training.update import send_info_to_central, send_update_to_central
from functions.processing.data import preprocess_into_train_test_and_eval_tensors
from functions.platforms.minio import get_object_data_and_metadata, create_or_update_object
from functions.training.model import local_model_training

import time
import os
# Refactored
def status_pipeline(
    task_file_lock: any,
    task_logger: any,
    task_minio_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any
):
    # Works
    status = send_info_to_central(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Status sending:' + str(status))
# Refactoroed
def data_pipeline(
    task_file_lock: any,
    task_logger: any,
    task_minio_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any
):
    cycle_start = time.time()
    # Check
    status = preprocess_into_train_test_and_eval_tensors(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )

    if status:
        workers_bucket = 'workers'
        worker_experiments_folder = os.environ.get('WORKER_ID') + '/experiments'
        worker_status_path = worker_experiments_folder + '/status'
        worker_status_object = get_object_data_and_metadata(
            logger = task_logger,
            minio_client = task_minio_client,
            bucket_name = workers_bucket,
            object_path = worker_status_path
        )
        worker_status = worker_status_object['data']

        experiment_folder_path = worker_experiments_folder + '/' + str(worker_status['experiment'])
        times_path = experiment_folder_path + '/times'

        times_object = get_object_data_and_metadata(
            logger = task_logger,
            minio_client = task_minio_client,
            bucket_name = workers_bucket,
            object_path = times_path
        )
        times = times_object['data']

        times[str(worker_status['cycle'])] = {
            'cycle-time-start':cycle_start,
            'cycle-time-end': 0,
            'cycle-total-seconds': 0
        }

        create_or_update_object(
            logger = task_logger,
            minio_client = task_minio_client,
            bucket_name = workers_bucket,
            object_path = times_path,
            data = times,
            metadata = {}
        )

    task_logger.info('Data preprocessing:' + str(status))
# Refactored
def model_pipeline(
    task_file_lock: any,
    task_logger: any,
    task_minio_client: any,
    task_mlflow_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any
): 
    # Check
    status = local_model_training(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        mlflow_client = task_mlflow_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Model training:' + str(status))
# Refactored
def update_pipeline(
    task_file_lock: any,
    task_logger: any,
    task_minio_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any
):
    # Check
    status = send_update_to_central(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Update sending:' + str(status))
