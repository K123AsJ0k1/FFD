import os
from functions.storage import store_file_data

# Created and works
def initilize_storage_templates(
    file_lock: any
):
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
            'total-ram-amount-bytes': 0.0,
            'available-ram-amount-bytes': 0.0,
            'total-disk-amount-bytes': 0.0,
            'available-disk-amount-bytes': 0.0
        },
        'function': {
            '1': {
                '1': { 
                    'name': 'initial-model-training',           
                    'time-seconds': 0.0,
                    'cpu-percentage': 0.0,
                    'ram-bytes': 0.0,
                    'disk-bytes': 0.0
                }
            }
        },
        'network': {
            '1': {
                '1': {
                    'name': 'sending-context',
                    'status-code': 0,
                    'processing-time-seconds': 0.0,
                    'elapsed-time-seconds': 0.0,
                    'cpu-percentage': 0.0,
                    'ram-bytes': 0.0,
                    'disk-bytes': 0.0
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
                    'ram-bytes': 0.0,
                    'disk-bytes': 0.0
                },
                '2': {
                    'name': 'model-testing',
                    'batches': 0,
                    'average-batch-size': 0,
                    'time-seconds': 0.0,
                    'cpu-percentage': 0.0,
                    'ram-bytes': 0.0,
                    'disk-bytes': 0.0
                },
                '3': {
                    'name': 'model-evaluation',
                    'batches': 0,
                    'average-batch-size': 0,
                    'time-seconds': 0.0,
                    'cpu-percentage': 0.0,
                    'ram-bytes': 0.0,
                    'disk-bytes': 0.0
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
                    'ram-bytes': 0.0,
                    'disk-bytes': 0.0
                }
            }
        }
    }

    paths = [
        'parameters/templates/model.txt',
        'parameters/templates/worker.txt',
        'status/templates/worker.txt',
        'metrics/templates/local.txt',
        'resources/templates/worker.txt'
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
    for template_path in paths:
        first_split = template_path.split('.')
        second_split = first_split[0].split('/')
        path_template = templates[second_split[0]][second_split[2]]
        template_folder_path = second_split[0] + '/templates'
        # This replaces 
        store_file_data(
            file_lock = file_lock,
            replace = False,
            file_folder_path = template_folder_path,
            file_path = template_path,
            data = path_template
        ) 
    os.environ['STATUS'] = 'initilized'