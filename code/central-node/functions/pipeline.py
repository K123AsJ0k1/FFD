from flask import current_app

import os 
import json

from functions.general import get_current_experiment_number
from functions.split import central_worker_data_split, split_data_between_workers
from functions.data import preprocess_into_train_test_and_evaluate_tensors
from functions.model import initial_model_training
from functions.update import send_context_to_workers
from functions.aggregation import update_global_model, evalute_global_model

# Refactored and works
def start_pipeline():
    current_experiment_number = get_current_experiment_number()
    central_status_path = 'status/experiment_' + str(current_experiment_number) + '/central.txt'
    if not os.path.exists(central_status_path):
        return False
    
    central_status = None
    with open(central_status_path, 'r') as f:
        central_status = json.load(f)

    central_status['start'] = True
    with open(central_status_path, 'w') as f:
        json.dump(central_status, f, indent=4) 
    return True
# Refactored
def data_pipeline(
    task_logger: any
):
    # Works
    status = central_worker_data_split(
        logger = task_logger
    )
    task_logger.info('Central-worker data split:' + str(status))
    # Works
    status = preprocess_into_train_test_and_evaluate_tensors(
        logger = task_logger
    )
    task_logger.info('Central pool preprocessing:' + str(status))
    # Check
    status = split_data_between_workers(
        logger = task_logger
    )
    task_logger.info('Worker data split:' + str(status))
# Created and works
def model_pipeline(
    task_logger: any
):  
    # Works
    status = initial_model_training(
        logger = task_logger
    )
    task_logger.info('Initial model training:' + str(status))
# Created
def update_pipeline(
    task_logger: any
):
    status = send_context_to_workers(
        logger = task_logger
    )
    task_logger.info('Worker context sending:' + str(status))
# Created
def aggregation_pipeline(
    task_logger: any
):
    status = update_global_model(
        logger = task_logger
    )
    task_logger.info('Updating global model:' + str(status))
    
    status = evalute_global_model(
        logger = task_logger
    )
    task_logger.info('Global model evaluation:' + str(status))

