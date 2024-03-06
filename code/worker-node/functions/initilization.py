from flask import current_app

import os
import json

# Created and works
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
        'storage/parameters/templates/model.txt',
        'storage/parameters/templates/worker.txt',
        'storage/status/templates/worker.txt',
        'storage/metrics/templates/local.txt',
        'storage/resources/templates/worker.txt'
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
        path_template = templates[second_split[1]][second_split[3]]
        if not os.path.exists(path):
            folder_path = 'storage/' + second_split[1] + '/templates'
            os.makedirs(folder_path, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(path_template , f, indent=4) 