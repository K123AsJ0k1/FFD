import numpy as np
import torch 
import os 
from collections import OrderedDict
import time
import psutil

from functions.model import FederatedLogisticRegression, evaluate
from functions.general import get_current_experiment_number, get_newest_model_updates, get_file_data, get_wanted_model
from functions.storage import store_metrics_and_resources, store_file_data
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
    file_lock: any,
    logger: any
) -> bool:
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    current_experiment_number = get_current_experiment_number()
    central_status_path = 'status/experiment_' + str(current_experiment_number) + '/central.txt'
    central_status = get_file_data(
        file_lock = file_lock,
        file_path = central_status_path
    )

    if central_status is None:
        return False
    
    if not central_status['start']:
        return False

    if central_status['complete']:
        return False

    if not central_status['sent']:
        return False

    if central_status['updated']:
        return False
    
    os.environ['STATUS'] = 'updating global model'
    logger.info('Updating global model')
    
    central_parameters_path = 'parameters/experiment_' + str(current_experiment_number) + '/central.txt'
    
    central_parameters = get_file_data(
        file_lock = file_lock,
        file_path = central_parameters_path
    )

    if central_parameters is None:
        return False

    available_updates, collective_sample_size = get_newest_model_updates(
        file_lock = file_lock,
        current_cycle = central_status['cycle']
    )
    # Could be reconsidered
    if not central_parameters['min-update-amount'] <= len(available_updates):
        return False
    
    new_global_model = model_fed_avg(
        updates = available_updates,
        total_sample_size = collective_sample_size 
    )
    update_model_path = 'models/experiment_' + str(current_experiment_number) + '/global_' + str(central_status['cycle']) + '_' + str(len(available_updates)) + '_' + str(collective_sample_size) + '.pth'
    
    store_file_data(
        file_lock = file_lock,
        replace = False,
        file_folder_path = '',
        file_path = update_model_path,
        data = new_global_model
    )
    
    central_status['collective-amount'] = collective_sample_size
    central_status['updated'] = True
    store_file_data(
        file_lock = file_lock,
        replace = True,
        file_folder_path = '',
        file_path = central_status_path,
        data = central_status
    )

    os.environ['STATUS'] = 'global model updated'
    logger.info('Global model updated')

    time_end = time.time()
    cpu_end = this_process.cpu_percent(interval=0.2)
    mem_end = psutil.virtual_memory().used 
    disk_end = psutil.disk_usage('.').used
    
    time_diff = (time_end - time_start) 
    cpu_diff = cpu_end - cpu_start 
    mem_diff = (mem_end - mem_start) # Bytes
    disk_diff = (disk_end - disk_start)

    resource_metrics = {
        'name': 'update-global-model',
        'time-seconds': round(time_diff,5),
        'cpu-percentage': cpu_diff,
        'ram-bytes': round(mem_diff,5),
        'disk-bytes': round(disk_diff,5)
    }

    status = store_metrics_and_resources(
        file_lock = file_lock,
        type = 'resources',
        subject = 'central',
        area = 'function',
        metrics = resource_metrics
    )

    return True
# Refactored and works
def evalute_global_model(
    file_lock: any,
    logger: any
):
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    current_experiment_number = get_current_experiment_number()
    central_status_path = 'status/experiment_' + str(current_experiment_number) + '/central.txt'
    
    central_status = get_file_data(
        file_lock = file_lock,
        file_path = central_status_path
    )

    if central_status is None:
        return False
    
    if not central_status['start']:
        return False
    
    if not central_status['updated']:
        return False
    
    if central_status['evaluated']:
        return False
    
    os.environ['STATUS'] = 'evaluating global model'
    logger.info('Evaluating global model')

    parameters_folder_path = 'parameters/experiment_' + str(current_experiment_number)
    central_parameters_path = parameters_folder_path + '/central.txt'
    model_parameters_path = parameters_folder_path + '/model.txt'

    central_parameters = get_file_data(
        file_lock = file_lock,
        file_path = central_parameters_path
    )

    if central_parameters is None:
        return False
    
    model_parameters = get_file_data(
        file_lock = file_lock,
        file_path = model_parameters_path
    )

    if model_parameters is None:
        return False
    
    current_global_model_parameters = get_wanted_model(
        file_lock = file_lock,
        experiment = current_experiment_number,
        subject = 'global',
        cycle = central_status['cycle']
    )
   
    lr_model = FederatedLogisticRegression(dim = model_parameters['input-size'])
    lr_model.apply_parameters(lr_model, current_global_model_parameters)

    evaluate(
        file_lock = file_lock,
        train_amount = central_status['collective-amount'],
        current_model = lr_model
    )

    global_metrics_path = 'metrics/experiment_' + str(current_experiment_number) + '/global.txt'
    
    global_metrics = get_file_data(
        file_lock = file_lock,
        file_path = global_metrics_path
    )

    evaluation_metrics = global_metrics[str(len(global_metrics))]

    succesful_metrics = 0
    thresholds = central_parameters['metric-thresholds']
    conditions = central_parameters['metric-conditions']
    for key,value in evaluation_metrics.items():
        if 'amount' in key:
            continue
        message = 'Metric ' + str(key)
        if conditions[key] == '>=' and thresholds[key] <= value:
            message = message + ' succeeded with ' + str(value) + str(conditions[key]) + str(thresholds[key])
            logger.info(message)
            succesful_metrics += 1
            continue
        if conditions[key] == '<=' and value <= thresholds[key]:
            message = message + ' succeeded with ' + str(value) + str(conditions[key]) + str(thresholds[key])
            logger.info(message)
            succesful_metrics += 1
            continue
        message = message + ' failed with ' + str(value) + str(conditions[key]) + str(thresholds[key])
        logger.info(message)
 
    central_status['evaluated'] = True
    if central_parameters['min-metric-success'] <= succesful_metrics or central_status['cycle'] == central_parameters['max-cycles']:
        message = 'Global model achieved ' + str(succesful_metrics) + '/' + str(central_parameters['min-metric-success']) + ' in ' + str(central_status['cycle']) + '/' + str(central_parameters['max-cycles'])
        logger.info(message)
        central_status['complete'] = True
        central_status['sent'] = False
        central_status['cycle'] = central_status['cycle'] + 1
    else: 
        message = 'Global model failed ' + str(succesful_metrics) + '/' + str(central_parameters['min-metric-success']) + ' in ' + str(central_status['cycle']) + '/' + str(central_parameters['max-cycles'])
        logger.info(message)
        central_status['worker-split'] = False
        central_status['sent'] = False
        central_status['updated'] = False
        central_status['evaluated'] = False
        central_status['worker-updates'] = 0
        central_status['cycle'] = central_status['cycle'] + 1

    central_resources_path = 'resources/experiment_' + str(current_experiment_number) + '/central.txt'
    central_resources = get_file_data(
        file_lock = file_lock,
        file_path = central_resources_path
    )
    # Potential info loss
    cycle_start = central_resources['general']['times'][str(central_status['cycle']-1)]['cycle-time-start']
    cycle_end = time.time()
    cycle_total = cycle_end-cycle_start
    central_resources['general']['times'][str(central_status['cycle']-1)]['cycle-time-end'] = cycle_end
    central_resources['general']['times'][str(central_status['cycle']-1)]['cycle-total-seconds'] = cycle_total
    if not central_status['complete']:
        central_resources['general']['times'][str(central_status['cycle'])] = {
            'cycle-time-start':time.time(),
            'cycle-time-end': 0,
            'cycle-total-seconds': 0
        }
    store_file_data(
        file_lock = file_lock,
        replace = True,
        file_folder_path = '',
        file_path = central_resources_path,
        data = central_resources
    )
        
    store_file_data(
        file_lock = file_lock,
        replace = True,
        file_folder_path = '',
        file_path = central_status_path,
        data = central_status
    )

    os.environ['STATUS'] = 'global model evaluated'
    logger.info('Global model evaluated')
    
    time_end = time.time()
    cpu_end = this_process.cpu_percent(interval=0.2)
    mem_end = psutil.virtual_memory().used 
    disk_end = psutil.disk_usage('.').used
    
    time_diff = (time_end - time_start) 
    cpu_diff = cpu_end - cpu_start 
    mem_diff = (mem_end - mem_start)
    disk_diff = (disk_end - disk_start)

    resource_metrics = {
        'name': 'evaluate-global-model',
        'time-seconds': round(time_diff,5),
        'cpu-percentage': cpu_diff,
        'ram-bytes': round(mem_diff,5),
        'disk-bytes': round(disk_diff,5)
    }

    status = store_metrics_and_resources(
        file_lock = file_lock,
        type = 'resources',
        subject = 'central',
        area = 'function',
        metrics = resource_metrics
    ) 

    return True