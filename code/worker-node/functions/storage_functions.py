from flask import current_app

import pandas as pd
import torch  
import os
import json

from collections import OrderedDict
 
# Created
def initilize_storage_templates():
    # Types: 0 = int, [] = list, 0.0 = float and {} = dict 
    model_parameters = {
        'seed': 0,
        'used-columns': [],
        'input-size': 0,
        'target-column': '',
        'scaled-columns': [],
        'learning-rate': 0.0,
        'sample-rate': 0.0,
        'optimizer': '',
        'epochs': 0
    }

    worker_parameters = {
        'sample-pool': 0,
        'data-augmentation': {
            'active': False,
            'sample-pool': 0,
            '1-0-ratio': 0.0
        },
        'eval-ratio': 0.0,
        'train-ratio': 0.0
    }
    # key format: worker id
    worker_status = {
        'id': 0,
        'central-address': '',
        'worker-address': '',
        'status': '',
        'stored': False,
        'preprocessed': False,
        'trained': False,
        'updated': False,
        'complete': False,
        'train-amount': 0,
        'test-amount':0,
        'eval-amount': 0,
        'cycle': 1
    }
    # key format: worker id and cycle
    local_metrics = {
        '1':{
            '1': {
                'train-amount': 0,
                'test-amount': 0,
                'eval-amount': 0,
                'true-positives': 0,
                'false-positives': 0,
                'true-negatives': 0,
                'false-negatives': 0,
                'recall': 0.0,
                'selectivity': 0.0,
                'precision': 0.0,
                'miss-rate': 0.0,
                'fall-out': 0.0,
                'balanced-accuracy': 0.0,
                'accuracy': 0.0
            }
        }
    }
    # key format: subject, cycle and id
    worker_resources = {
        'general': {
            'physical-cpu-amount': 0,
            'total-cpu-amount': 0,
            'min-cpu-frequency-mhz': 0.0,
            'max-cpu-frequency-mhz': 0.0,
            'total-ram-amount-megabytes': 0.0,
            'available-ram-amount-megabytes': 0.0,
            'total-disk-amount-megabytes': 0.0,
            'available-disk-amount-megabytes': 0.0
        },
        'function': {
            '1': {
                '1': { 
                    'name': 'initial-model-training',           
                    'time-seconds': 0.0,
                    'cpu-percentage': 0.0,
                    'ram-megabytes': 0.0,
                    'disk-megabytes': 0.0
                }
            }
        },
        'network': {
            '1': {
                '1': {
                    'name': 'sending-context',
                    'time-seconds': 0.0,
                    'cpu-percentage': 0.0,
                    'ram-megabytes': 0.0,
                    'disk-megabytes': 0.0
                }
            }
        },
        'training': {
            '1': {
                '1': {
                    'name': 'model-training',
                    'epochs': 0,
                    'batches': 0,
                    'average-batch-size': 0,
                    'time-seconds': 0.0,
                    'cpu-percentage': 0.0,
                    'ram-megabytes': 0.0,
                    'disk-megabytes': 0.0
                },
                '2': {
                    'name': 'model-testing',
                    'batches': 0,
                    'average-batch-size': 0,
                    'time-seconds': 0.0,
                    'cpu-percentage': 0.0,
                    'ram-megabytes': 0.0,
                    'disk-megabytes': 0.0
                },
                '3': {
                    'name': 'model-evaluation',
                    'batches': 0,
                    'average-batch-size': 0,
                    'time-seconds': 0.0,
                    'cpu-percentage': 0.0,
                    'ram-megabytes': 0.0,
                    'disk-megabytes': 0.0
                }
            }
        },
        'inference': {
            '1': {
                '1': {
                    'name': 'model-prediction',
                    'sample-amount': 0,
                    'time-seconds': 0.0,
                    'cpu-percentage': 0.0,
                    'ram-megabytes': 0.0,
                    'disk-megabytes': 0.0
                }
            }
        }
    }

    paths = [
        'parameters/model.txt',
        'parameters/worker.txt',
        'status/worker.txt',
        'metrics/local.txt',
        'resources/worker.txt'
    ]

    templates = {
        'parameters': {
            'model': model_parameters,
            'worker': worker_parameters
        },
        'status': {
            'worker': worker_status
        },
        'metrics': {
            'local': local_metrics
        },
        'resources': {
            'worker': worker_resources
        }
    }

    os.environ['STATUS'] = 'initilizing'
    for path in paths:
        first_split = path.split('.')
        second_split = first_split[0].split('/')
        path_template = templates[second_split[0]][second_split[1]]
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump(path_template , f, indent=4) 
# Created and works
def get_current_experiment_number():
    parameter_files = os.listdir('status')
    highest_experiment_number = 0
    for file in parameter_files:
        if not '.txt' in file:
            experiment_number = int(file.split('_')[1])    
            if highest_experiment_number < experiment_number:
                highest_experiment_number = experiment_number
    return highest_experiment_number
# Refactor
def store_training_context(
    parameters: any,
    global_model: any,
    df_data: list,
    df_columns: list
) -> any:
    # Separate training artifacts will have the following folder format of experiment_(int)
    current_experiment_number = get_current_experiment_number()
    existing_worker_status_path = 'status/experiment_' + str(current_experiment_number) + '/worker.txt'
    if os.path.exists(existing_worker_status_path):
        worker_status = None
        with open(existing_worker_status_path, 'r') as f:
            worker_status = json.load(f)
        if not worker_status['complete']:
            return False
    
    if not worker_status['id'] == int(worker_parameters['id']):
        return 'wrong id'
    
    if worker_status['stored'] and not worker_status['updated']:
        return 'ongoing jobs'
    
    if global_parameters == None:
        worker_status['address'] = worker_parameters['address']
        worker_status['completed'] = True
        worker_status['cycle'] = worker_parameters['cycle']
    else:
        worker_status['address'] = worker_parameters['address']
        worker_status['trained'] = False
        worker_status['updated'] = False
        worker_status['completed'] = False
        worker_status['columns'] = worker_parameters['columns']
        worker_status['train-test-ratio'] = worker_parameters['train-test-ratio']
        worker_status['cycle'] = worker_parameters['cycle']
    
        global_parameters_path = 'logs/global_parameters.txt'
        with open(global_parameters_path, 'w') as f:
            json.dump(global_parameters, f, indent=4)
    
    os.environ['STATUS'] = 'storing'
    
    global_model_path = 'models/global_' + str(worker_parameters['cycle']) + '.pth'
    
    weights = global_model['weights']
    bias = global_model['bias']
    
    formated_parameters = OrderedDict([
        ('linear.weight', torch.tensor(weights,dtype=torch.float32)),
        ('linear.bias', torch.tensor(bias,dtype=torch.float32))
    ])
    
    torch.save(formated_parameters, global_model_path)
    if not worker_data == None:
        worker_data_path = 'data/sample_' + str(worker_parameters['cycle']) + '.csv'
        worker_df = pd.DataFrame(worker_data)
        worker_df.to_csv(worker_data_path, index = False)
        worker_status['preprocessed'] = False

    worker_status['stored'] = True
    with open(worker_status_path, 'w') as f:
        json.dump(worker_status, f, indent=4)

    os.environ['STATUS'] = 'stored'

    return 'stored'
# Refactor
def store_local_metrics(
   metrics: any
) -> bool:
    #worker_status_path = 'logs/worker_status.txt'
    #if not os.path.exists(worker_status_path):
    #    return False
    #worker_status = None
    #with open(worker_status_path, 'r') as f:
    #    worker_status = json.load(f)

    new_key = len(worker_status['local-metrics'])
    worker_status['local-metrics'][str(new_key)] = metrics
    with open(worker_status_path, 'w') as f:
        json.dump(worker_status, f, indent=4) 
    return True