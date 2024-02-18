from flask import current_app

import torch 
import os
import json

'''
training status format:
- entry: dict
   - parameters: dict
      - pooling: bool
      - preprocess: bool
      - training: bool
      - splitting: bool
      - update: bool
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
# Refactored and works
def initilize_training_status():
    training_status_path = 'logs/training_status.txt'
    if os.path.exists(training_status_path):
        return False
   
    os.environ['STATUS'] = 'initilizing'

    training_status = {
        'parameters': {
            'data-split': False,
            'preprocessed': False,
            'trained': False,
            'worker-split': False,
            'sent': False,
            'updated': False,
            'evaluated': False,
            'worker-updates': 0,
            'cycle': 0,
            'columns': None,
            'global-metrics': []
        },
        'workers': {}
    }
   
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
    
    training_status = None
    if not os.path.exists(training_status_path):
        return False
    
    with open(training_status_path, 'r') as f:
        training_status = json.load(f)

    if worker_id == None:
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
        
        local_metrics = []
        if -1 < duplicate_id:
            local_metrics = training_status['workers'][str(duplicate_id)]['local-metrics']
            del training_status['workers'][str(duplicate_id)]

        training_status['workers'][str(smallest_missing_id)] = {
            'address': worker_ip,
            'status': worker_status,
            'stored': False,
            'preprocessed': False,
            'training': False,
            'updated': False,
            'cycle': 0,
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
# Refactpred and works
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
# Refactored
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

    if not training_status['parameters']['sent']:
        return False

    model_path = 'models/worker_' + str(worker_id) + '_' + str(cycle) + '_' + str(train_size) + '.pth'
    if os.path.exists(model_path):
        return False
    
    torch.save(local_model, model_path)

    index = 0
    for worker in training_status['workers']:
        if worker['id'] == worker_id:
            training_status['workers'][index]['status'] = 'complete'

    training_status['parameters']['updates'] = training_status['parameters']['updates'] + 1
    with open(training_status_path, 'w') as f:
        json.dump(training_status, f, indent=4) 

    return True