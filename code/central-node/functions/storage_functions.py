from flask import current_app

import torch 
import os
import json
import copy
import pandas as pd
import psutil

from collections import OrderedDict

# Refactored and works
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

    central_parameters = {
        'sample-pool': 0,
        'data-augmentation': {
            'active': False,
            'sample-pool': 0,
            '1-0-ratio': 0.0
        },
        'eval-ratio': 0.0,
        'train-ratio': 0.0,
        'min-update-amount': 0,
        'max-cycles': 0,
        'min-metric-success': 0,
        'metric-thresholds': {
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
        },
        'metric-conditions': {
            'true-positives': '>=',
            'false-positives': '<=',
            'true-negatives': '>=', 
            'false-negatives': '<=',
            'recall': '>=',
            'selectivity': '>=',
            'precision': '>=',
            'miss-rate': '<=',
            'fall-out': '<=',
            'balanced-accuracy': '>=',
            'accuracy': '>='
        }
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

    central_status = {
        'start': False,
        'data-split': False,
        'preprocessed': False,
        'worker-split': False,
        'trained': False,
        'sent': False,
        'updated': False,
        'evaluated': False,
        'complete': False,
        'train-amount': 0,
        'test-amount': 0,
        'eval-amount': 0,
        'worker-updates': 0,
        'cycle': 1,
    }
    # key format: worker id
    worker_status = {
        '1': {
            'id': 1,
            'central-address': '',
            'worker-address': '',
            'status': '',
            'stored': False,
            'preprocessed': False,
            'trained': False,
            'updated': False,
            'train-amount': 0,
            'test-amount':0,
            'cycle': 1
        }
    }
    # key format: cycle
    global_metrics = {
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
    central_resources = {
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
    # Key format worker id
    worker_resources = {
        '1': {
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
                        'name': 'model-training',           
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
                        'name': 'sending-update',
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
                        'batch-size': 0,
                        'time-seconds': 0.0,
                        'cpu-percentage': 0.0,
                        'ram-megabytes': 0.0,
                        'disk-megabytes': 0.0
                    },
                    '2': {
                        'name': 'model-testing',
                        'batches': 0,
                        'batch-size': 0,
                        'time-seconds': 0.0,
                        'cpu-percentage': 0.0,
                        'ram-megabytes': 0.0,
                        'disk-megabytes': 0.0
                    },
                    '3': {
                        'name': 'model-evaluation',
                        'batches': 0,
                        'batch-size':0, 
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
    }

    paths = [
        'parameters/model.txt',
        'parameters/central.txt',
        'parameters/worker.txt',
        'status/central.txt',
        'status/workers.txt',
        'metrics/global.txt',
        'metrics/local.txt',
        'resources/central.txt',
        'resources/workers.txt'
    ]

    templates = {
        'parameters': {
            'model': model_parameters,
            'central': central_parameters,
            'worker': worker_parameters
        },
        'status': {
            'central': central_status,
            'workers': worker_status
        },
        'metrics': {
            'global': global_metrics,
            'local': local_metrics
        },
        'resources': {
            'central': central_resources,
            'workers': worker_resources
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
    parameter_files = os.listdir('parameters')
    highest_experiment_number = 0
    for file in parameter_files:
        if not '.txt' in file:
            experiment_number = int(file.split('_')[1])    
            if highest_experiment_number < experiment_number:
                highest_experiment_number = experiment_number
    return highest_experiment_number
# Refactored and works
def store_training_context(
    parameters: any,
    df_data: list,
    df_columns: list
) -> bool:
    # Separate training artifacts will have the following folder format of experiment_(int)
    current_experiment_number = get_current_experiment_number()
    existing_central_status_path = 'status/experiment_' + str(current_experiment_number) + '/central.txt'
    if os.path.exists(existing_central_status_path):
        central_status = None
        with open(existing_central_status_path, 'r') as f:
            central_status = json.load(f)
        if not central_status['complete']:
            return False

    new_folder_id = current_experiment_number + 1

    template_paths = [
        'parameters/model.txt',
        'parameters/central.txt',
        'parameters/worker.txt',
        'status/central.txt',
        'status/workers.txt',
        'metrics/global.txt',
        'metrics/local.txt',
        'resources/central.txt',
        'resources/workers.txt'
    ]

    for path in template_paths:
        first_split = path.split('.')
        second_split = first_split[0].split('/')

        stored_template = None
        with open(path, 'r') as f:
            stored_template = json.load(f)
        
        file_path = second_split[0] + '/experiment_' + str(new_folder_id) + '/' + second_split[1] + '.txt'
        if second_split[0] == 'parameters':
            given_parameters = parameters[second_split[1]]
            modified_template = copy.deepcopy(stored_template)
            for key in stored_template.keys():
                modified_template[key] = given_parameters[key]
            stored_template = copy.deepcopy(modified_template)
        if second_split[0] == 'resources' and second_split[1] == 'central':
            stored_template = {
                'general': {
                    'physical-cpu-amount': psutil.cpu_count(logical=False),
                    'total-cpu-amount': psutil.cpu_count(logical=True),
                    'min-cpu-frequency-mhz': psutil.cpu_freq().min,
                    'max-cpu-frequency-mhz': psutil.cpu_freq().max,
                    'total-ram-amount-megabytes': psutil.virtual_memory().total / (1024 ** 2),
                    'available-ram-amount-megabytes': psutil.virtual_memory().free / (1024 ** 2),
                    'total-disk-amount-megabytes': psutil.disk_usage('.').total / (1024 ** 2),
                    'available-disk-amount-megabytes': psutil.disk_usage('.').free / (1024 ** 2)
                },
                'function': {},
                'network': {},
                'training': {},
                'inference': {}
            }
        if (second_split[0] == 'metrics' and (second_split[1] == 'global' or second_split[1] == 'local') 
            or (second_split[0] == 'status' and second_split[1] == 'workers')
            or (second_split[0] == 'resources' and second_split[1] == 'workers')):
            stored_template = {}
        
        if not os.path.exists(file_path):
            directory_path = second_split[0] + '/experiment_' + str(new_folder_id)
            os.makedirs(directory_path, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(stored_template, f, indent=4)

    file_path = 'data/experiment_' + str(new_folder_id) + '/source.csv'
    if not os.path.exists(file_path):
        directory_path = 'data/experiment_' + str(new_folder_id)
        os.makedirs(directory_path, exist_ok=True)
        source_df = pd.DataFrame(df_data, columns = df_columns)
        source_df.to_csv(file_path)
    return True
# Refactored
def store_metrics_and_resources( 
   type: str,
   subject: str,
   area: str,
   metrics: any
) -> bool:
    current_experiment_number = get_current_experiment_number()
    stored_data = None
    storage_path = None
    if type == 'metrics':
        if subject == 'global':
            storage_path = 'metrics/experiment_' + str(current_experiment_number) + '/global.txt'
            if not os.path.exists(storage_path):
                return False
        
            stored_data = None
            with open(storage_path, 'r') as f:
                stored_data = json.load(f)

            new_key = len(stored_data) + 1
            stored_data[str(new_key)] = metrics
    if type == 'resources':
        central_status_path = 'status/experiment_' + str(current_experiment_number) + '/central.txt'
        if not os.path.exists(central_status_path):
            return False
        
        central_status = None
        with open(central_status_path, 'r') as f:
            central_status = json.load(f)

        if subject == 'central':
            storage_path = 'resources/experiment_' + str(current_experiment_number) + '/central.txt'
            if not os.path.exists(storage_path):
                return False
            
            stored_data = None
            with open(storage_path, 'r') as f:
                stored_data = json.load(f)

            if not str(central_status['cycle']) in stored_data[area]:
                stored_data[area][str(central_status['cycle'])] = {}
            new_key = len(stored_data[area][str(central_status['cycle'])]) + 1
            stored_data[area][str(central_status['cycle'])][str(new_key)] = metrics
    #print(stored_data)
    with open(storage_path, 'w') as f:
        json.dump(stored_data, f, indent=4) 
    
    return True
# refactored
def store_worker(
    address: str,
    status: any,
    metrics: any
) -> any:
    current_experiment_number = get_current_experiment_number()
    status_folder_path = 'status/experiment_' + str(current_experiment_number)
    
    central_status_path = status_folder_path + '/central.txt'
    if not os.path.exists(central_status_path):
        return False
    
    central_status = None
    with open(central_status_path, 'r') as f:
        central_status = json.load(f)

    worker_status_path = status_folder_path + '/workers.txt'
    if not os.path.exists(worker_status_path):
        return False
    
    worker_status = None
    with open(worker_status_path, 'r') as f:
        worker_status = json.load(f)

    local_metrics_path = 'metrics/experiment_' + str(current_experiment_number) + '/local.txt'
    if not os.path.exists(local_metrics_path):
        return False
    
    local_metrics = None
    with open(local_metrics_path, 'r') as f:
        local_metrics = json.load(f)

    if worker_status['id'] == 0:
        # When worker isn't registered either due to being new or failure restart
        duplicate_id = -1
        used_keys = []
        
        for worker_key in worker_status.keys():
            worker_metadata = worker_status[worker_key]
            if worker_metadata['address'] == status['address']:
                duplicate_id = int(worker_key)
            used_keys.append(int(worker_key))
            
        set_of_used_keys = set(used_keys)
        smallest_missing_id = 0
        while smallest_missing_id in set_of_used_keys:
            smallest_missing_id += 1

        old_local_metrics = None
        
        if -1 < duplicate_id:
            old_local_metrics = local_metrics[str(duplicate_id)]
            del worker_status[str(duplicate_id)]
            del local_metrics[str(duplicate_id)]

        old_worker_data_path = 'data/experiment_' + str(current_experiment_number) + '/worker_' + str(duplicate_id) + '_' + str(central_status['cycle']) + '.csv'
        if os.path.exists(old_worker_data_path):
            new_worker_data_path = 'worker_' + str(smallest_missing_id) + '_' + str(central_status['cycle']) + '.csv'
            os.rename(old_worker_data_path,new_worker_data_path)

        modified_status = copy.deepcopy(status)
        modified_status['id'] = str(smallest_missing_id)
        modified_status['address'] = address
        modified_status['cycle'] = central_status['cycle']

        worker_status[str(smallest_missing_id)] = modified_status
        local_metrics[str(smallest_missing_id)] = old_local_metrics

        with open(worker_status_path, 'w') as f:
            json.dump(worker_status, f, indent=4)

        with open(local_metrics_path, 'w') as f:
            json.dump(local_metrics, f, indent=4)
        
        return 'registered', worker_status, local_metrics
    else:
        worker_metadata = worker_status[str(worker_status['id'])]
        action = ''
        if worker_metadata['address'] == address:
            # When worker is already registered and address has stayed the same
            modified_metadata = copy.deepcopy(status)
            worker_status[str(worker_status['id'])] = modified_metadata
            local_metrics[str(worker_status['id'])] = metrics['local']
            resource_metrics[str(worker_status['id'])] = metrics['resource']
            action = 'checked'
        else:
            # When worker id has stayed the same, but address has changed due to load balancing
            modified_metadata = copy.deepcopy(status)
            modified_metadata['address'] = address
            worker_status[str(worker_status['id'])] = modified_metadata
            local_metrics[str(worker_status['id'])] = metrics
            action = 'rerouted'
            
        with open(worker_status_path, 'w') as f:
            json.dump(worker_status, f, indent=4)
            
        with open(local_metrics_path, 'w') as f:
            json.dump(local_metrics, f, indent=4)

        return action, worker_status, None
# Refactor
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

    if not training_status['parameters']['start']:
        return False

    if training_status['parameters']['complete']:
        return False

    if not training_status['parameters']['sent']:
        return False
    
    model_path = 'models/worker_' + str(worker_id) + '_' + str(cycle) + '_' + str(train_size) + '.pth'
    
    formatted_model = OrderedDict([
        ('linear.weight', torch.tensor(local_model['weights'],dtype=torch.float32)),
        ('linear.bias', torch.tensor(local_model['bias'],dtype=torch.float32))
    ])
    
    torch.save(formatted_model, model_path)
    for worker_key in training_status['workers'].keys():
        if worker_key == str(worker_id):
            training_status['workers'][worker_key]['status'] = 'complete'

    training_status['parameters']['worker-updates'] = training_status['parameters']['worker-updates'] + 1
    with open(training_status_path, 'w') as f:
        json.dump(training_status, f, indent=4) 

    return True