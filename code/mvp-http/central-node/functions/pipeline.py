from functions.general import get_current_experiment_number, get_file_data
from functions.split import central_worker_data_split, split_data_between_workers
from functions.data import preprocess_into_train_test_and_evaluate_tensors
from functions.model import initial_model_training
from functions.update import send_context_to_workers
from functions.aggregation import update_global_model, evalute_global_model
from functions.storage import store_file_data, store_metrics_and_resources
import time
# Refactored and works
def start_pipeline(
    file_lock: any
):
    current_experiment_number = get_current_experiment_number()
    central_status_path = 'status/experiment_' + str(current_experiment_number) + '/central.txt'
    central_status = get_file_data(
        file_lock = file_lock,
        file_path = central_status_path
    )

    if central_status is None:
        return False
    
    if central_status['start'] and not central_status['complete']:
        return False

    central_resources_path = 'resources/experiment_' + str(current_experiment_number) + '/central.txt'
    central_resources = get_file_data(
        file_lock = file_lock,
        file_path = central_resources_path
    )
    
    central_resources['general']['times'][str(central_status['cycle'])]['cycle-time-start'] = time.time()
    store_file_data(
        file_lock = file_lock,
        replace = True,
        file_folder_path = '',
        file_path = central_resources_path,
        data = central_resources
    )
    
    central_status['start'] = True
    store_file_data(
        file_lock = file_lock,
        replace = True,
        file_folder_path = '',
        file_path = central_status_path,
        data = central_status
    )

    return True
# Refactored and works
def data_pipeline(
    task_file_lock: any,
    task_logger: any
):
    # Works
    status = central_worker_data_split(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Central-worker data split:' + str(status))
    # Works
    status = preprocess_into_train_test_and_evaluate_tensors(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Central pool preprocessing:' + str(status))
# Refactored and works
def model_pipeline(
    task_file_lock: any,
    task_logger: any
):  
    # Works
    status = initial_model_training(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Initial model training:' + str(status))
# Refactored and works
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
# Refactored and works
def aggregation_pipeline(
    task_file_lock: any,
    task_logger: any
):
    # Check
    status = update_global_model(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Updating global model:' + str(status))
    # Check
    status = evalute_global_model(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Global model evaluation:' + str(status))

