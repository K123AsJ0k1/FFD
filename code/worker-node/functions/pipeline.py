from flask import current_app
import requests

import torch  
import os
import json

#from functions.data import *
#from functions.model import *
#from functions.storage import *
 
# Created
def data_pipeline(
    task_logger: any,
):
    status = preprocess_into_train_and_test_tensors(
        logger = task_logger
    )
    task_logger.info('Data preprocessing:' + str(status))
# Created
def model_pipeline(
    task_logger: any
): 
    status = local_model_training(
        logger = task_logger
    )
    task_logger.info('Model training:' + str(status))
# Created
def update_pipeline(
    task_logger: any
):
    # Check
    status = send_info_to_central(
        logger = task_logger
    )
    task_logger.info('Status sending:' + str(status))
    #status = send_update(
    #    logger = task_logger
    #)
    #task_logger.info('Update sending:' + str(status))