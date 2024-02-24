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
      - evaluated: bool
      - complete
      - worker-updates: int
      - cycle: int
      - columns: list
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
            'complete': False,
            'worker-updates': 0,
            'cycle': 0, 
            'columns': None,
            'global-metrics': {}
        },
        'workers': {}
    }
   
    with open(training_status_path, 'w') as f:
        json.dump(training_status, f, indent=4) 
    return True
# refactored and works
def store_worker_status(
    worker_address: str,
    worker_status: any
) -> any:
    training_status_path = 'logs/training_status.txt'
    
    training_status = None
    if not os.path.exists(training_status_path):
        return False
    
    with open(training_status_path, 'r') as f:
        training_status = json.load(f)

    if worker_status['id'] == None:
        duplicate_id = -1
        used_keys = []
        
        for worker_key in training_status['workers'].keys():
            worker_metadata = training_status['workers'][worker_key]
            if worker_metadata['address'] == worker_status['address']:
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
            'address': worker_address,
            'status': worker_status,
            'stored': False,
            'preprocessed': False,
            'trained': False,
            'updated': False,
            'cycle': 0,
            'local-metrics': local_metrics
        }
        with open(training_status_path, 'w') as f:
            json.dump(training_status, f, indent=4)
        
        return smallest_missing_id, worker_address, 'registered'
    else:
        worker_metadata = training_status['workers'][str(worker_status['id'])]
        if worker_metadata['address'] == worker_address:
            # When worker is already registered and address has stayed the same
            # Add stored
            worker_metadata['status'] = worker_status['status']
            worker_metadata['preprocessed'] = worker_status['preprocessed']
            worker_metadata['trained'] = worker_status['trained']
            worker_metadata['updated'] = worker_status['updated']
            worker_metadata['cycle'] = worker_status['cycle']
            worker_metadata['local-metrics'] = worker_status['local-metrics']
            training_status['workers'][str(worker_status['id'])] = worker_metadata

            with open(training_status_path, 'w') as f:
                json.dump(training_status, f, indent=4)
            return worker_status['id'], worker_address, 'checked'
        else:
            # When worker id has stayed the same, but address has changed due to load balancing
            worker_metadata['status'] = worker_status['status']
            worker_metadata['address'] = worker_address
            worker_metadata['preprocessed'] = worker_status['preprocessed']
            worker_metadata['trained'] = worker_status['trained']
            worker_metadata['updated'] = worker_status['updated']
            worker_metadata['cycle'] = worker_status['cycle']
            worker_metadata['local-metrics'] = worker_status['local-metrics']
            training_status['workers'][str(worker_status['id'])] = worker_metadata
            with open(training_status_path, 'w') as f:
                json.dump(training_status, f, indent=4)
            return worker_status['id'], worker_address, 'rerouted'
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

    highest_key = 0
    for id in training_status['parameters']['global-metrics']:
        if highest_key < int(id):
            highest_key = id
    
    training_status['parameters']['global-metrics'][str(highest_key)] = metrics
    with open(training_status_path, 'w') as f:
        json.dump(training_status, f, indent=4) 
    
    return True
# Refactored and works
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

    if training_status['parameters']['complete']:
        return False

    if not training_status['parameters']['sent']:
        return False

    model_path = 'models/worker_' + str(worker_id) + '_' + str(cycle) + '_' + str(train_size) + '.pth'
    torch.save(local_model, model_path)
    # Fix the inconsistent string worker id
    for worker_key in training_status['workers'].keys():
        if worker_key == str(worker_id):
            training_status['workers'][worker_key]['status'] = 'complete'

    training_status['parameters']['worker-updates'] = training_status['parameters']['worker-updates'] + 1
    with open(training_status_path, 'w') as f:
        json.dump(training_status, f, indent=4) 

    return True