from flask import current_app

import os 

from functions.storage import store_file_data

# Refactored and works
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
        'collective-amount': 0,
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
            'complete': False,
            'train-amount': 0,
            'test-amount':0,
            'eval-amount': 0,
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
                        'status-code': 0,
                        'processing-time-seconds': 0.0,
                        'elapsed-time-seconds': 0.0,
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
        'parameters/templates/model.txt',
        'parameters/templates/central.txt',
        'parameters/templates/worker.txt',
        'status/templates/central.txt',
        'status/templates/workers.txt',
        'metrics/templates/global.txt',
        'metrics/templates/local.txt',
        'resources/templates/central.txt',
        'resources/templates/workers.txt'
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
    
    for template_path in paths:
        first_split = template_path.split('.')
        second_split = first_split[0].split('/')
        path_template = templates[second_split[0]][second_split[2]]
        template_folder_path = second_split[0] + '/templates'
        store_file_data(
            file_lock = file_lock,
            replace = False,
            file_folder_path = template_folder_path,
            file_path = template_path,
            data = path_template
        )