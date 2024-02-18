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
- workers: dict
    - id: dict
        - address: str
        - status: str
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
# Refactored and works
def send_status_to_central(
    logger: any, 
    central_address: str
):
    payload = {
        'status': os.environ.get('STATUS'),
        'id': int(os.environ.get('ID'))
    }
    json_payload = json.dumps(payload) 
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
        os.environ['ID'] = str(given_data['id'])
    except Exception as e:
        logger.error('Status sending error:', e) 
# Refactored
def store_training_context(
    global_parameters: any,
    worker_parameters: any,
    global_model: any,
    worker_data: any
) -> bool:
    if not os.environ.get('ID') == worker_parameters['id']:
        return 'wrong id'
    worker_parameters_path = 'logs/worker_parameters.txt'
    
    if os.path.exists(worker_parameters_path):
        WORKER_PARAMETERS = None
        with open(worker_parameters_path, 'r') as f:
            WORKER_PARAMETERS = json.load(f)
        if not WORKER_PARAMETERS['updated']:
            return os.environ.get('STATUS')
        
    with open(worker_parameters_path, 'w') as f:
        json.dump(worker_parameters, f)

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

    WORKER_PARAMETERS['stored'] = True
    with open(worker_parameters_path, 'w') as f:
        json.dump(WORKER_PARAMETERS, f)

    return True
# Refactored
def preprocess_into_train_and_test_tensors() -> bool:
    worker_parameters_path = 'logs/worker_parameters.txt'
    WORKER_PARAMETERS = None
    with open(worker_parameters_path, 'r') as f:
        WORKER_PARAMETERS = json.load(f)
    
    if WORKER_PARAMETERS['preprocessed']:
        return False
    
    worker_data_path = 'data/used_data_' + str(WORKER_PARAMETERS['cycle']) + '.csv'
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
    preprocessed_df.columns = WORKER_PARAMETERS['columns']
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
        train_size = WORKER_PARAMETERS['train-test-ratio'], 
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

    WORKER_PARAMETERS['preprocessed'] = True
    with open(worker_parameters_path, 'w') as f:
        json.dump(WORKER_PARAMETERS, f)

    return True
# Created
def store_local_metrics(
   metrics: any
) -> bool:
   training_metrics_path = 'logs/training_metrics.txt'
   if not os.path.exists(training_metrics_path):
      return False
   training_metrics = None
   with open(training_metrics_path, 'r') as f:
      training_status = json.load(f)
   training_metrics.append(metrics)
   with open(training_metrics, 'w') as f:
      json.dump(training_status, f, indent=4) 
   return True