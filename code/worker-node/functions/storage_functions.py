from flask import current_app

import pandas as pd
import torch  
import os
import json

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
        'updated': False,
        'columns': None,
        'train-test-ratio': 0,
        'cycle': 0,
        'local-metrics': []
    }
   
    with open(worker_status_path, 'w') as f:
        json.dump(worker_status, f, indent=4)

    os.environ['STATUS'] = 'waiting' 
    return True
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
    
    if not worker_status['id'] == int(worker_parameters['id']):
        return 'wrong id'
    
    if worker_status['stored'] and not worker_status['updated']:
        return 'ongoing jobs'
    
    worker_status['address'] = worker_parameters['address']
    worker_status['columns'] = worker_parameters['columns']
    worker_status['train-test-ratio'] = worker_parameters['train-test-ratio']
    worker_status['cycle'] = worker_parameters['cycle']
    
    global_parameters_path = 'logs/global_parameters.txt'
    with open(global_parameters_path, 'w') as f:
        json.dump(global_parameters, f, indent=4)
    
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
        json.dump(worker_status, f, indent=4)

    os.environ['STATUS'] = 'stored'

    return 'stored'
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