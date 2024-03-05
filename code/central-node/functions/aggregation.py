from flask import current_app

import numpy as np
import pandas as pd
import torch 
import os 
import json
from collections import OrderedDict

from functions.storage import *
from functions.model import *

# Refactored and works 
def model_fed_avg(
    updates: any,
    total_sample_size: int    
) -> any:
    weights = []
    biases = []
    for update in updates:
        parameters = update['parameters']
        worker_sample_size = update['samples']
        
        worker_weights = np.array(parameters['linear.weight'].tolist()[0])
        worker_bias = np.array(parameters['linear.bias'].tolist()[0])
        
        adjusted_worker_weights = worker_weights * (worker_sample_size/total_sample_size)
        adjusted_worker_bias = worker_bias * (worker_sample_size/total_sample_size)
        
        weights.append(adjusted_worker_weights.tolist())
        biases.append(adjusted_worker_bias)
    
    FedAvg_weight = [np.sum(weights,axis = 0)]
    FedAvg_bias = [np.sum(biases, axis = 0)]

    updated_global_model = OrderedDict([
        ('linear.weight', torch.tensor(FedAvg_weight,dtype=torch.float32)),
        ('linear.bias', torch.tensor(FedAvg_bias,dtype=torch.float32))
    ])
    return updated_global_model
# Refactor
def update_global_model(
    logger: any
) -> bool:
    training_status_path = 'logs/training_status.txt'
    if not os.path.exists(training_status_path):
        return False
   
    training_status = None
    with open(training_status_path, 'r') as f:
        training_status = json.load(f)

    if not training_status['parameters']['start']:
        return False

    if training_status['parameters']['complete']:
        return False

    if not training_status['parameters']['sent']:
        return False

    if training_status['parameters']['updated']:
        return False

    if training_status['parameters']['worker-updates'] < central_parameters['min-update-amount']:
        return False
    
    files = os.listdir('models')
    available_updates = []
    collective_sample_size = 0
    for file in files:
        if 'worker' in file:
            first_split = file.split('.')
            second_split = first_split[0].split('_')
            cycle = int(second_split[2])
            sample_size = int(second_split[3])

            if cycle == training_status['parameters']['cycle']:
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
    update_model_path = 'models/global_' + str(training_status['parameters']['cycle'] + 1) + '_' + str(len(available_updates)) + '_' + str(collective_sample_size) + '.pth'
    torch.save(new_global_model, update_model_path)

    with open(training_status_path, 'r') as f:
        training_status = json.load(f)
    training_status['parameters']['updated'] = True
    with open(training_status_path, 'w') as f:
         json.dump(training_status, f, indent=4)
    return True
# Refactor
def evalute_global_model(
    logger: any
):
    training_status_path = 'logs/training_status.txt'
    if not os.path.exists(training_status_path):
        return False
    
    training_status = None
    with open(training_status_path, 'r') as f:
        training_status = json.load(f)

    if not training_status['parameters']['start']:
        return False

    if not training_status['parameters']['updated']:
        return False

    if training_status['parameters']['evaluated']:
        return False
    
    models_folder_path = 'models'
    if not os.path.exists(models_folder_path):
        return False
    
    files = os.listdir(models_folder_path)
    if len(files) == 0:
        return False  
    
    current_global_model = ''
    highest_key = 0
    for file in files:
        first_split = file.split('.')
        second_split = first_split[0].split('_')
        name = str(second_split[0])
        if name == 'global':
            cycle = int(second_split[1])
            if highest_key < cycle:
                highest_key = cycle
                current_global_model = 'models/' + file 
    
    eval_tensor_path = 'tensors/eval.pt'
    given_parameters = torch.load(current_global_model)
    
    lr_model = FederatedLogisticRegression(dim = global_parameters['input-size'])
    lr_model.apply_parameters(lr_model, given_parameters)

    eval_tensor = torch.load(eval_tensor_path)
    eval_loader = DataLoader(eval_tensor, 64)

    test_metrics = test(
        model = lr_model, 
        test_loader = eval_loader
    )
    
    status = store_global_metrics(
        metrics = test_metrics
    )
    
    succesful_metrics = 0
    thresholds = central_parameters['metric-thresholds']
    conditions = central_parameters['metric-conditions']
    for key,value in test_metrics.items():
        logger.info('Metric ' + str(key) + ' with threshold ' + str(thresholds[key]) + ' and condition ' + str(conditions[key]))
        if conditions[key] == '>=' and thresholds[key] <= value:
            logger.info('Passed with ' + str(value))
            succesful_metrics += 1
            continue
        if conditions[key] == '<=' and value <= thresholds[key]:
            logger.info('Passed with ' + str(value))
            succesful_metrics += 1
            continue
        logger.info('Failed with ' + str(value))

    with open(training_status_path, 'r') as f:
        training_status = json.load(f)
    
    training_status['parameters']['evaluated'] = True
    if central_parameters['min-metric-success'] <= succesful_metrics or training_status['parameters']['cycle'] == central_parameters['max-cycles']:
        training_status['parameters']['complete'] = True
        training_status['parameters']['sent'] = False
        training_status['parameters']['cycle'] = training_status['parameters']['cycle'] + 1
    else: 
        training_status['parameters']['worker-split'] = False
        training_status['parameters']['sent'] = False
        training_status['parameters']['updated'] = False
        training_status['parameters']['evaluated'] = False
        training_status['parameters']['worker-updates'] = 0
        training_status['parameters']['cycle'] = training_status['parameters']['cycle'] + 1
    with open(training_status_path, 'w') as f:
         json.dump(training_status, f, indent=4) 

    return True