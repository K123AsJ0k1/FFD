from flask import current_app
import requests

import numpy as np
import pandas as pd
import torch  
import os
import json

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from collections import OrderedDict

'''
worker status format:
- status dict
    - id: dict
    - address: str
    - stored: str
    - preprocessed: bool
    - training: bool
    - updating: bool
    - columns: list
    - train-test-ratio: float
    - cycle: int
    - local metrics: list
        - metrics: dict
            - confusion list
            - recall: int
            - selectivity: int
            - precision: int
            - miss-rate: int
            - fall-out: int
            - balanced-accuracy: int 
            - accuracy: int
'''
# Created
def initilize_worker_status():
    worker_status_path = 'logs/worker_status.txt'
    if os.path.exists(worker_status_path):
        return False
   
    os.environ['STATUS'] = 'initilizing'
    worker_status = {
        'id': None,
        'address': None,
        'stored': False,
        'preprocessed': False,
        'trained': False,
        'updating': False,
        'columns': None,
        'train-test-ratio': 0,
        'cycle': 0,
        'local-metrics': []
    }
   
    with open(worker_status_path, 'w') as f:
        json.dump(worker_status, f, indent=4) 
    return True
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
# Refactored
def store_training_context(
    global_parameters: any,
    worker_parameters: any,
    global_model: any,
    worker_data: any
) -> any:
    worker_status_path = 'logs/worker_status.txt'
    if not os.path.exists(worker_status_path):
        return 'missing initilization'
    
    worker_status = None
    with open(worker_status_path, 'r') as f:
        worker_status = json.load(f)

    if not worker_status['id'] == worker_parameters['id']:
        return 'wrong id'
    
    if not worker_status['updated']:
        return 'ongoing jobs'
    
    worker_status['address'] = worker_parameters['address']
    worker_status['columns'] = worker_parameters['columns']
    worker_status['train-test-ratio'] = worker_parameters['train-test-ratio']
    worker_status['cycle'] = worker_parameters['cycle']
    
    global_parameters_path = 'logs/global_parameters.txt'
    with open(global_parameters_path, 'w') as f:
        json.dump(global_parameters, f)
    
    os.environ['STATUS'] = 'storing'
    
    global_model_path = 'models/global_model_' + str(worker_parameters['cycle']) + '.pth'
    worker_data_path = 'data/used_data_' + str(worker_parameters['cycle']) + '.csv'
    
    weights = global_model['weights']
    bias = global_model['bias']
    
    formated_parameters = OrderedDict([
        ('linear.weight', torch.tensor(weights,dtype=torch.float32)),
        ('linear.bias', torch.tensor(bias,dtype=torch.float32))
    ])
    
    torch.save(formated_parameters, global_model_path)
       
    worker_df = pd.DataFrame(worker_data)
    worker_df.to_csv(worker_data_path, index = False)

    worker_status['stored'] = True
    with open(worker_status_path, 'w') as f:
        json.dump(worker_status, f)

    os.environ['STATUS'] = 'stored'

    return 'stored'
# Refactored
def preprocess_into_train_and_test_tensors() -> bool:
    worker_status_path = 'logs/worker_status.txt'
    if not os.path.exists(worker_status_path):
        return False
    
    worker_status = None
    with open(worker_status_path, 'r') as f:
        worker_status = json.load(f)

    if worker_status['preprocessed']:
        return False
    
    worker_data_path = 'data/used_data_' + str(worker_status['cycle']) + '.csv'
    if not os.path.exists(worker_data_path):
       return False
    
    global_parameters_path = 'logs/global_parameters.txt'
    train_tensor_path = 'tensors/train.pt'
    test_tensor_path = 'tensors/test.pt'
    
    os.environ['STATUS'] = 'preprocessing'
    
    GLOBAL_PARAMETERS = None
    with open(global_parameters_path, 'r') as f:
        GLOBAL_PARAMETERS = json.load(f) 

    preprocessed_df = pd.read_csv(worker_data_path)
    preprocessed_df.columns = worker_status['columns']
    preprocessed_df = preprocessed_df[GLOBAL_PARAMETERS['used-columns']]
    for column in GLOBAL_PARAMETERS['scaled-columns']:
        mean = preprocessed_df[column].mean()
        std_dev = preprocessed_df[column].std()
        preprocessed_df[column] = (preprocessed_df[column] - mean)/std_dev

    X = preprocessed_df.drop(GLOBAL_PARAMETERS['target-column'], axis = 1).values
    y = preprocessed_df[GLOBAL_PARAMETERS['target-column']].values
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        train_size = worker_status['train-test-ratio'], 
        random_state = GLOBAL_PARAMETERS['seed']
    )

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)

    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)
    
    train_tensor = TensorDataset(
        torch.tensor(X_train), 
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_tensor = TensorDataset(
        torch.tensor(X_test), 
        torch.tensor(y_test, dtype=torch.float32)
    )
    
    torch.save(train_tensor,train_tensor_path)
    torch.save(test_tensor,test_tensor_path)

    worker_status['preprocessed'] = True
    with open(worker_status_path, 'w') as f:
        json.dump(worker_status, f)

    os.environ['STATUS'] = 'preprocessed'

    return True
# Refactored
def store_local_metrics(
   metrics: any
) -> bool:
    worker_status_path = 'logs/worker_status.txt'
    if not os.path.exists(worker_status_path):
        return False
    worker_status = None
    with open(worker_status_path, 'r') as f:
        worker_status = json.load(f)
    worker_status['local-metrics'].append(metrics)
    with open(worker_status_path, 'w') as f:
        json.dump(worker_status, f, indent=4) 
    return True