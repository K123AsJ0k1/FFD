from flask import current_app
import requests

import torch  
import os
import json

from functions.data_functions import *
from functions.model_functions import *
from functions.storage_functions import *

# Refactored and works
def send_status_to_central(
    logger: any, 
    central_address: str
) -> bool:
    worker_status_path = 'logs/worker_status.txt'
    if not os.path.exists(worker_status_path):
        return False

    worker_status = None
    with open(worker_status_path, 'r') as f:
        worker_status = json.load(f)

    worker_status['status'] = os.environ.get('STATUS')
    json_payload = json.dumps(worker_status) 
    address = central_address + '/status'
    try:
        response = requests.post(
            url = address,
            json = json_payload,
            headers = {
               'Content-type':'application/json', 
               'Accept':'application/json'
            }
        )
        given_data = json.loads(response.text)
        worker_status['id'] = given_data['id']
        with open(worker_status_path, 'w') as f:
            json.dump(worker_status, f, indent=4) 
        return True
    except Exception as e:
        logger.error('Status sending error:', e) 
        return False
# Created    
def send_update(
    logger: any, 
    central_address: str
):  
    worker_status_path = 'logs/worker_status.txt'
    if not os.path.exists(worker_status_path):
        return False
    worker_status = None
    with open(worker_status_path, 'r') as f:
        worker_status = json.load(f)

    if not worker_status['stored'] or not worker_status['preprocessed'] or not worker_status['trained']:
        return False

    if worker_status['updated']:
        return False

    os.environ['STATUS'] = 'updating'

    local_model_path = 'models/local_model_' + str(worker_status['cycle']) + '.pth'
    local_model = torch.load(local_model_path)

    formatted_local_model = {
      'weights': local_model['linear.weight'].numpy().tolist(),
      'bias': local_model['linear.bias'].numpy().tolist()
    }

    train_tensor = torch.load('tensors/train.pt')
    
    payload = {
        'worker-id': worker_status['worker-id'],
        'local-model': formatted_local_model,
        'cycle': worker_status['cycle'],
        'train-size': len(train_tensor)
    }
    
    json_payload = json.dumps(payload)
    central_url = central_address + '/update'
    try:
        response = requests.post(
            url = central_url, 
            json = json_payload,
            headers = {
                'Content-type':'application/json', 
                'Accept':'application/json'
            }
        )
        if response.status_code == 200:
            worker_status['updated'] = True
            with open(worker_status_path, 'w') as f:
                json.dump(worker_status, f)
            os.environ['STATUS'] = 'updated'
    except Exception as e:
        logger.error('Status sending error:', e)
# Created
def run_training_pipeline(
    logger: any
):
    status = preprocess_into_train_and_test_tensors()
    status = local_model_training()