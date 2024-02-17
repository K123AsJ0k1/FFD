from flask import current_app

import numpy as np
import pandas as pd
import torch 
import os
import json
import requests
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

'''
training status format:
- entry: dict
   - parameters: dict
      - cycle: int
      - global metrics: list
         - metrics: dict
            - confusion list
            - recall: int
            - selectivity: int
            - precision: int
            - miss-rate: int
            - fall-out: int
            - balanced-accuracy: int 
            - accuracy: int
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
# Created and works
def initilize_training_status():
    training_status_path = 'logs/training_status.txt'
    if os.path.exists(training_status_path):
        return False
   
    training_status = {
        'parameters': {
            'cycle': 0,
            'global-metrics': []
        },
        'workers': {}
    }
   
    with open(training_status_path, 'w') as f:
        json.dump(training_status, f, indent=4) 
    return True
# Works
def central_worker_data_split() -> bool:
    central_pool_path = 'data/central_pool.csv'
    worker_pool_path = 'data/worker_pool.csv'

    if os.path.exists(central_pool_path) or os.path.exists(worker_pool_path):
        return False
    
    CENTRAL_PARAMETERS = current_app.config['CENTRAL_PARAMETERS']
    WORKER_PARAMETERS = current_app.config['WORKER_PARAMETERS']

    data_path = 'data/formated_fraud_detection_data.csv'
    source_df = pd.read_csv(data_path)

    splitted_data_df = source_df.drop('step', axis = 1)
    
    central_data_pool = splitted_data_df.sample(n =  CENTRAL_PARAMETERS['sample-pool'])
    central_indexes = central_data_pool.index.tolist()
    splitted_data_df.drop(central_indexes)
    worker_data_pool = splitted_data_df.sample(n =  WORKER_PARAMETERS['sample-pool'])

    central_data_pool.to_csv(central_pool_path, index = False)    
    worker_data_pool.to_csv(worker_pool_path, index = False)
    return True
# Works
def preprocess_into_train_test_and_evaluate_tensors() -> bool:
    GLOBAL_PARAMETERS = current_app.config['GLOBAL_PARAMETERS']
    CENTRAL_PARAMETERS = current_app.config['CENTRAL_PARAMETERS']
    
    central_pool_path = 'data/central_pool.csv'

    if not os.path.exists(central_pool_path):
        return False
    
    train_tensor_path = 'tensors/train.pt'
    test_tensor_path = 'tensors/test.pt'
    eval_tensor_path = 'tensors/eval.pt'

    if os.path.exists(train_tensor_path) or os.path.exists(test_tensor_path) or os.path.exists(eval_tensor_path):
        return False

    central_data_df = pd.read_csv(central_pool_path)
    
    preprocessed_df = central_data_df[GLOBAL_PARAMETERS['used-columns']]
    for column in GLOBAL_PARAMETERS['scaled-columns']:
        mean = preprocessed_df[column].mean()
        std_dev = preprocessed_df[column].std()
        preprocessed_df[column] = (preprocessed_df[column] - mean)/std_dev

    X = preprocessed_df.drop(GLOBAL_PARAMETERS['target-column'], axis = 1).values
    y = preprocessed_df[GLOBAL_PARAMETERS['target-column']].values
        
    X_train_test, X_eval, y_train_test, y_eval = train_test_split(
        X, 
        y, 
        train_size = CENTRAL_PARAMETERS['train-eval-ratio'], 
        random_state = GLOBAL_PARAMETERS['seed']
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_test, 
        y_train_test, 
        train_size = CENTRAL_PARAMETERS['train-test-ratio'], 
        random_state = GLOBAL_PARAMETERS['seed']
    )

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    X_eval = np.array(X_eval, dtype=np.float32)

    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)
    y_eval = np.array(y_eval, dtype=np.int32)
    
    train_tensor = TensorDataset(
        torch.tensor(X_train), 
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_tensor = TensorDataset(
        torch.tensor(X_test), 
        torch.tensor(y_test, dtype=torch.float32)
    )
    eval_tensor = TensorDataset(
        torch.tensor(X_eval), 
        torch.tensor(y_eval, dtype=torch.float32)
    )

    torch.save(train_tensor,train_tensor_path)
    torch.save(test_tensor,test_tensor_path)
    torch.save(eval_tensor,eval_tensor_path)
    
    return True
# Works
def split_data_between_workers(
    worker_amount: int
) -> any:
    worker_pool_path = 'data/worker_pool.csv'

    if not os.path.exists(worker_pool_path):
        return False
    
    worker_pool_df = pd.read_csv(worker_pool_path)
    
    worker_df = worker_pool_df.sample(frac = 1)
    worker_dfs = np.array_split(worker_df, worker_amount)
    
    data_list = []
    index = 1
    for assigned_df in worker_dfs:
        assigned_df.to_csv('data/worker_' + str(index) + '.csv', index = False)
        data_list.append(assigned_df.values.tolist())
        index = index + 1

    return data_list, worker_pool_df.columns.tolist()     
# Created
def store_global_metrics(
   metrics: any
) -> bool:
    training_status_path = 'logs/training_status.txt'
    if not os.path.exists(training_status_path):
        return False
    training_status = None
    with open(training_status_path, 'r') as f:
        training_status = json.load(f)
    training_status['parameters']['global-metrics'].append(metrics)
    with open(training_status_path, 'w') as f:
        json.dump(training_status, f, indent=4) 
    return True
# refactored and works
def store_worker_status(
    worker_id: int,
    worker_ip: str,
    worker_status: str
) -> any:
    training_status_path = 'logs/training_status.txt'
    #print(type(worker_id))
    training_status = None
    if not os.path.exists(training_status_path):
        return False
    
    with open(training_status_path, 'r') as f:
        training_status = json.load(f)

    if worker_id == -1:
        duplicate_id = -1
        used_keys = []
        
        for worker_key in training_status['workers'].keys():
            worker_metadata = training_status['workers'][worker_key]
            if worker_metadata['address'] == worker_ip:
                duplicate_id = int(worker_key)
            used_keys.append(int(worker_key))
            
        set_of_used_keys = set(used_keys)
        smallest_missing_id = 0
        while smallest_missing_id in set_of_used_keys:
            smallest_missing_id += 1
        
        local_metrics = {}
        if -1 < duplicate_id:
            local_metrics = training_status['workers'][str(duplicate_id)]['local-metrics']
            del training_status['workers'][str(duplicate_id)]

        training_status['workers'][str(smallest_missing_id)] = {
            'address': worker_ip,
            'status': worker_status,
            'local-metrics': local_metrics
        }
        with open(training_status_path, 'w') as f:
            json.dump(training_status, f, indent=4)
        
        return smallest_missing_id, 'registered'
    else:
        worker_metadata = training_status['workers'][str(worker_id)]
        if worker_metadata['address'] == worker_ip:
            # When worker is already registered and address has stayed the same
            training_status['workers'][str(worker_id)]['status'] = worker_status
            with open(training_status_path, 'w') as f:
                json.dump(training_status, f, indent=4)
            return worker_id, 'checked'
        else:
            # When worker id has stayed the same, but address has changed due to load balancing
            training_status['workers'][str(worker_id)]['status'] = worker_status
            training_status['workers'][str(worker_id)]['address'] = worker_ip
            with open(training_status_path, 'w') as f:
                json.dump(training_status, f, indent=4)
            return worker_id, 'rerouted'
# Works
def store_update(
    worker_id: str,
    local_model: any,
    cycle: int,
    train_size: int
) -> bool:
    training_status_path = 'logs/training_status.txt'
   
    training_status = None
    if not os.path.exists(training_status_path):
      return False
    
    with open(training_status_path, 'r') as f:
        training_status = json.load(f)

    model_path = 'models/worker_' + str(worker_id) + '_' + str(cycle) + '_' + str(train_size) + '.pth'
    if os.path.exists(model_path):
        return False
    
    torch.save(local_model, model_path)

    index = 0
    for worker in training_status['workers']:
        if worker['id'] == worker_id:
            training_status['workers'][index]['status'] = 'complete'

    with open(training_status_path, 'w') as f:
        json.dump(training_status, f, indent=4) 

    return True
# Refactored
def send_context_to_workers():
   GLOBAL_PARAMETERS = current_app.config['GLOBAL_PARAMETERS']
   WORKER_PARAMETERS = current_app.config['WORKER_PARAMETERS']
   
   training_status_path = 'logs/training_status.txt'
   if not os.path.exists(training_status_path):
      return False
   
   training_status = None
   with open(training_status_path, 'r') as f:
      training_status = json.load(f)
   
   global_model_path = 'models/global_model_' + str(training_status['parameters']['cycle']) + '.pth'
   if not os.path.exists(global_model_path):
      return False

   global_model = torch.load(global_model_path)
   data_list, columns = split_data_between_workers(
      worker_amount = len(training_status['workers'])
   )
   
   formatted_global_model = {
      'weights': global_model['linear.weight'].numpy().tolist(),
      'bias': global_model['linear.bias'].numpy().tolist()
   }
   
   index = 0
   for dict in training_status['workers']:
      worker_address = 'http://' + dict['address'] + ':7500/context'
      
      worker_parameters = WORKER_PARAMETERS.copy()
      worker_parameters['address'] = dict['address']
      worker_parameters['worker-id'] = dict['id']
      worker_parameters['status'] = dict['status']
      worker_parameters['cycle'] = training_status['parameters']['cycle']
      worker_parameters['columns'] = columns
      
      payload = {
         'worker-id': dict['id'],
         'global-parameters': GLOBAL_PARAMETERS,
         'worker-parameters': worker_parameters,
         'global-model': formatted_global_model,
         'worker-data': data_list[index]
      }

      json_payload = json.dumps(payload) 

      try:
         response = requests.post(
            url = worker_address, 
            json = json_payload,
            headers = {
               'Content-type':'application/json', 
               'Accept':'application/json'
            }
         )
      except Exception as e:
         current_app.logger.error('Context sending error')
         current_app.logger.error(e)

      index = index + 1

