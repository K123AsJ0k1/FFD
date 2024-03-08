from flask import current_app

import numpy as np
import torch 
import os 
import json
from collections import OrderedDict
import time
import psutil

from functions.model import FederatedLogisticRegression, evaluate
from functions.general import get_current_experiment_number, get_newest_model_updates, get_current_global_model
from functions.storage import store_metrics_and_resources
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
# Refactored and works
def update_global_model(
    logger: any
) -> bool:
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    storage_folder_path = 'storage'

    current_experiment_number = get_current_experiment_number()
    central_status_path = storage_folder_path + '/status/experiment_' + str(current_experiment_number) + '/central.txt'
    if not os.path.exists(central_status_path):
        return False
    
    central_status = None
    with open(central_status_path, 'r') as f:
        central_status = json.load(f)

    if not central_status['start']:
        return False

    if central_status['complete']:
        return False

    if not central_status['sent']:
        return False

    if central_status['updated']:
        return False
    
    central_parameters_path = storage_folder_path + '/parameters/experiment_' + str(current_experiment_number) + '/central.txt'
    if not os.path.exists(central_parameters_path):
        return False

    central_parameters = None
    with open(central_parameters_path, 'r') as f:
        central_parameters = json.load(f)

    available_updates, collective_sample_size = get_newest_model_updates(
        current_cycle = central_status['cycle']
    )

    if not central_parameters['min-update-amount'] <= len(available_updates):
        return False
    
    new_global_model = model_fed_avg(
        updates = available_updates,
        total_sample_size = collective_sample_size 
    )
    update_model_path = storage_folder_path + '/models/experiment_' + str(current_experiment_number) + '/global_' + str(central_status['cycle']) + '_' + str(len(available_updates)) + '_' + str(collective_sample_size) + '.pth'
    torch.save(new_global_model, update_model_path)

    central_status['collective-amount'] = collective_sample_size
    central_status['updated'] = True
    with open(central_status_path, 'w') as f:
         json.dump(central_status, f, indent=4)

    time_end = time.time()
    cpu_end = this_process.cpu_percent(interval=0.2)
    mem_end = psutil.virtual_memory().used 
    disk_end = psutil.disk_usage('.').used
    
    time_diff = (time_end - time_start) 
    cpu_diff = cpu_end - cpu_start 
    mem_diff = (mem_end - mem_start) / (1024 ** 2) 
    disk_diff = (disk_end - disk_start) / (1024 ** 2)

    resource_metrics = {
        'name': 'update-global-model',
        'time-seconds': round(time_diff,5),
        'cpu-percentage': cpu_diff,
        'ram-megabytes': round(mem_diff,5),
        'disk-megabytes': round(disk_diff,5)
    }

    status = store_metrics_and_resources(
        type = 'resources',
        subject = 'central',
        area = 'function',
        metrics = resource_metrics
    )

    return True
# Refactored and works
def evalute_global_model(
    logger: any
):
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    storage_folder_path = 'storage'

    current_experiment_number = get_current_experiment_number()
    central_status_path = storage_folder_path + '/status/experiment_' + str(current_experiment_number) + '/central.txt'
    if not os.path.exists(central_status_path):
        return False    

    central_status = None
    with open(central_status_path, 'r') as f:
        central_status = json.load(f)

    if not central_status['start']:
        return False

    if not central_status['updated']:
        return False

    if central_status['evaluated']:
        return False

    parameters_folder_path = storage_folder_path  + '/parameters/experiment_' + str(current_experiment_number)
    central_parameters_path = parameters_folder_path + '/central.txt'
    
    if not os.path.exists(central_parameters_path):
        return False    

    central_parameters = None
    with open(central_parameters_path, 'r') as f:
        central_parameters = json.load(f)
    
    model_parameters_path = parameters_folder_path + '/model.txt'
    if not os.path.exists(model_parameters_path):
        return False    

    model_parameters = None
    with open(model_parameters_path, 'r') as f:
        model_parameters = json.load(f)

    current_global_model_parameters = get_current_global_model()
    lr_model = FederatedLogisticRegression(dim = model_parameters['input-size'])
    lr_model.apply_parameters(lr_model, current_global_model_parameters)

    evaluate(
        train_amount = central_status['collective-amount'],
        current_model = lr_model
    )

    global_metrics_path = storage_folder_path + '/metrics/experiment_' + str(current_experiment_number) + '/global.txt'
    if not os.path.exists(global_metrics_path):
        return False    

    global_metrics = None
    with open(global_metrics_path, 'r') as f:
        global_metrics = json.load(f)

    evaluation_metrics = global_metrics[str(len(global_metrics))]

    succesful_metrics = 0
    thresholds = central_parameters['metric-thresholds']
    conditions = central_parameters['metric-conditions']
    for key,value in evaluation_metrics.items():
        if 'amount' in key:
            continue
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
 
    central_status['evaluated'] = True
    if central_parameters['min-metric-success'] <= succesful_metrics or central_status['cycle'] == central_parameters['max-cycles']:
        logger.info('Global model has passed ' + str(succesful_metrics) + ' metrics, stopping training')
        central_status['complete'] = True
        central_status['sent'] = False
        central_status['cycle'] = central_status['cycle'] + 1
    else: 
        logger.info('Global model pased only' + str(succesful_metrics) + ' metrics, continouing training')
        central_status['worker-split'] = False
        central_status['sent'] = False
        central_status['updated'] = False
        central_status['evaluated'] = False
        central_status['worker-updates'] = 0
        central_status['cycle'] = central_status['cycle'] + 1
    
    time_end = time.time()
    cpu_end = this_process.cpu_percent(interval=0.2)
    mem_end = psutil.virtual_memory().used 
    disk_end = psutil.disk_usage('.').used
    
    time_diff = (time_end - time_start) 
    cpu_diff = cpu_end - cpu_start 
    mem_diff = (mem_end - mem_start) / (1024 ** 2) 
    disk_diff = (disk_end - disk_start) / (1024 ** 2)

    resource_metrics = {
        'name': 'evaluate-global-model',
        'time-seconds': round(time_diff,5),
        'cpu-percentage': cpu_diff,
        'ram-megabytes': round(mem_diff,5),
        'disk-megabytes': round(disk_diff,5)
    }

    status = store_metrics_and_resources(
        type = 'resources',
        subject = 'central',
        area = 'function',
        metrics = resource_metrics
    ) 

    with open(central_status_path, 'w') as f:
        json.dump(central_status, f, indent=4)

    return True