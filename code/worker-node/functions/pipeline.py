from flask import current_app

from functions.data import preprocess_into_train_test_and_eval_tensors
from functions.model import local_model_training
from functions.update import send_info_to_central, send_update_to_central
# Created
def status_pipeline(
    task_logger: any
):
    # Works
    status = send_info_to_central(
        logger = task_logger
    )
    task_logger.info('Status sending:' + str(status))
# Refactored and works
def data_pipeline(
    task_logger: any
):
    # Works
    status = preprocess_into_train_test_and_eval_tensors(
        logger = task_logger
    )
    task_logger.info('Data preprocessing:' + str(status))
# Refactored and works
def model_pipeline(
    task_logger: any
): 
    # Works
    status = local_model_training(
        logger = task_logger
    )
    task_logger.info('Model training:' + str(status))
# Refactored
def update_pipeline(
    task_logger: any
):
    status = send_update_to_central(
        logger = task_logger
    )
    task_logger.info('Update sending:' + str(status))