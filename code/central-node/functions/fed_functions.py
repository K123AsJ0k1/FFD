from flask import current_app

import numpy as np
import pandas as pd
import torch 
import os
import json
import requests
from collections import OrderedDict

from functions.storage_functions import *
from functions.model_functions import *

# Refactored
def send_context_to_workers(
    logger: any
) -> bool:
    training_status_path = 'logs/training_status.txt'
    if not os.path.exists(training_status_path):
        return False
   
    training_status = None
    with open(training_status_path, 'r') as f:
        training_status = json.load(f)

    if not training_status['parameters']['training']:
        return False
    
    if not training_status['parameters']['worker-splitting']:
        return False

    if training_status['parameters']['sending']:
        return False

    GLOBAL_PARAMETERS = current_app.config['GLOBAL_PARAMETERS']
    WORKER_PARAMETERS = current_app.config['WORKER_PARAMETERS']
   
    global_model_path = 'models/global_model_' + str(training_status['parameters']['cycle']) + '.pth'
    if not os.path.exists(global_model_path):
        return False
    
    os.environ['STATUS'] = 'sending'

    global_model = torch.load(global_model_path)
    formatted_global_model = {
        'weights': global_model['linear.weight'].numpy().tolist(),
        'bias': global_model['linear.bias'].numpy().tolist()
    }

    payload_status = {}
    for worker_key in training_status['workers'].keys():
        worker_metadata = training_status['workers'][worker_key]
        
        if not worker_metadata['status'] == 'waiting':
            continue

        data_path = 'data/worker_' + worker_key + '_' + str(training_status['parameters']['cycle']) + '.csv'
        worker_df = pd.read_csv(data_path)
        worker_data = worker_df.values.tolist()
        worker_url = 'http://' + worker_metadata['address'] + ':7500/context'
        
        sent_worker_parameters = {
            'id': worker_metadata['id'],
            'address': worker_metadata['address'],
            'columns': training_status['parameters']['columns'],
            'cycle': training_status['parameters']['cycle'],
            'train-test-ratio': WORKER_PARAMETERS['train-test-ratio']
        }
        
        payload = {
            'global-parameters': GLOBAL_PARAMETERS,
            'worker-parameters': sent_worker_parameters,
            'global-model': formatted_global_model,
            'worker-data': worker_data
        }

        json_payload = json.dumps(payload) 
        try:
            response = requests.post(
                url = worker_url, 
                json = json_payload,
                headers = {
                    'Content-type':'application/json', 
                    'Accept':'application/json'
                }
            )
            payload_status[worker_key] = {
                'address': worker_metadata['address'],
                'response':response.status_code
            }
        except Exception as e:
            logger.error('Context sending error' + e)
    
    successes = 0
    for worker_key in payload_status.keys():
        worker_data = payload_status[worker_key]
        if not worker_data['response'] == 200: 
            store_worker_status(
                worker_id = int(worker_key),
                worker_ip = worker_data['address'],
                worker_status = 'failure'
            )
            continue
        successes = successes + 1
    
    if not GLOBAL_PARAMETERS['min-update-amount'] <= successes:
        return False

    training_status['parameters']['sending'] = True
    with open(training_status_path, 'w') as f:
        json.dump(training_status, f, indent=4)
    
    os.environ['STATUS'] = 'waiting updates'

    return True
# Refactored
def model_fed_avg(
    updates: any,
    total_sample_size: int    
) -> any:
    weights = []
    biases = []
    for update in updates:
        parameters = update['parameters']
        worker_sample_size = update['samples']

        worker_weights = np.array(parameters['weights'][0])
        worker_bias = np.array(parameters['bias'])
        
        adjusted_worker_weights = worker_weights * (worker_sample_size/total_sample_size)
        adjusted_worker_bias = worker_bias * (worker_sample_size/total_sample_size)
        
        weights.append(adjusted_worker_weights.tolist())
        biases.append(adjusted_worker_bias)

    FedAvg_weight = [np.sum(weights,axis = 0)]
    FedAvg_bias = np.sum(biases, axis = 0)

    updated_global_model = OrderedDict([
        ('linear.weight', torch.tensor(FedAvg_weight,dtype=torch.float32)),
        ('linear.bias', torch.tensor(FedAvg_bias,dtype=torch.float32))
    ])
    return updated_global_model
# Refactored
def update_global_model():
    training_status_path = 'logs/training_status.txt'
    if not os.path.exists(training_status_path):
        return False
    
    training_status = None
    with open(training_status_path, 'r') as f:
        training_status = json.load(f)

    update_model_path = 'models/global_model_' + str(training_status['parameters']['cycle']) + '.pth'
    if os.path.exists(update_model_path):
        return True
    
    files = os.listdir('models')
    available_updates = []
    collective_sample_size = 0
    for file in files:
        if 'worker' in file:
            first_split = file.split('.')
            second_split = first_split[0].split('_')
            worker_id = int(second_split[1])
            cycle = int(second_split[2])
            sample_size = int(second_split[3])

            if cycle == training_status['parameters']['cycle']:
                if training_status['workers'][worker_id-1]['status'] == 'complete':
                    local_model_path = 'models/' + file
                    available_updates.append({
                        'parameters': torch.load(local_model_path),
                        'samples': sample_size
                    })
                    collective_sample_size = collective_sample_size + sample_size

    new_global_model = model_fed_avg(
        updates = available_updates,
        total_sample_size = collective_sample_size 
    )

    torch.save(new_global_model, update_model_path)
    training_status['parameters']['cycle'] = training_status['parameters']['cycle'] + 1 
    with open(training_status_path, 'w') as f:
         json.dump(training_status, f, indent=4) 
    return True
# Refactored
def evalute_global_model():
    training_status_path = 'logs/training_status.txt'
    if not os.path.exists(training_status_path):
        return False
    
    training_status = None
    with open(training_status_path, 'r') as f:
        training_status = json.load(f)
    
    GLOBAL_PARAMETERS = current_app.config['GLOBAL_PARAMETERS']
    
    global_model_path = 'models/global_model_' + str(training_status['parameters']['cycle']) + '.pth'
    if not os.path.exists(global_model_path):
        return False
    
    eval_tensor_path = 'tensors/evaluation.pt'
    if not os.path.exists(eval_tensor_path):
        return False
    
    given_parameters = torch.load(global_model_path)
    
    lr_model = FederatedLogisticRegression(dim = GLOBAL_PARAMETERS['input-size'])
    lr_model.apply_parameters(lr_model, given_parameters)

    eval_tensor = torch.load(eval_tensor_path)
    eval_loader = DataLoader(eval_tensor, 64)

    test_metrics = test(
        model = lr_model, 
        test_loader = eval_loader
    )

    store_global_metrics(
        metrics = test_metrics
    )

    return True