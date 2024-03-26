from functions.training.update import send_info_to_central

# Refactor
def status_pipeline(
    task_file_lock: any,
    task_logger: any,
    task_minio_client: any,
    task_prometheus_registry: any,
    task_prometheus_metrics: any
):
    # 
    status = send_info_to_central(
        file_lock = task_file_lock,
        logger = task_logger,
        minio_client = task_minio_client,
        prometheus_registry = task_prometheus_registry,
        prometheus_metrics = task_prometheus_metrics
    )
    task_logger.info('Status sending:' + str(status))
'''
# Refactored and works
def data_pipeline(
    task_file_lock: any,
    task_logger: any
):
    cycle_start = time.time()
    # Check
    status = preprocess_into_train_test_and_eval_tensors(
        file_lock = task_file_lock,
        logger = task_logger
    )

    if status:
        current_experiment_number = get_current_experiment_number()
        worker_status_path = 'status/experiment_' + str(current_experiment_number) + '/worker.txt'
    
        worker_status = get_file_data(
            file_lock = task_file_lock,
            file_path = worker_status_path
        )

        worker_resources_path = 'resources/experiment_' + str(current_experiment_number) + '/worker.txt'
        worker_resources = get_file_data(
            file_lock = task_file_lock,
            file_path = worker_resources_path
        )

        worker_resources['general']['times'][str(worker_status['cycle'])] = {
            'cycle-time-start':cycle_start,
            'cycle-time-end': 0,
            'cycle-total-seconds': 0
        }

        store_file_data(
            file_lock = task_file_lock,
            replace = True,
            file_folder_path = '',
            file_path = worker_resources_path,
            data = worker_resources
        )

    task_logger.info('Data preprocessing:' + str(status))
# Refactored and works
def model_pipeline(
    task_file_lock: any,
    task_logger: any
): 
    # Check
    status = local_model_training(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Model training:' + str(status))
# Refactored
def update_pipeline(
    task_file_lock: any,
    task_logger: any
):
    # Check
    status = send_update_to_central(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Update sending:' + str(status))
'''