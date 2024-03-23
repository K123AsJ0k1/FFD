from functions.platforms.minio import get_object_list, create_or_update_object, check_object
import psutil
from datetime import datetime

# Refactored and works
def initilize_minio(
    logger: any,
    minio_client: any
):
    templates = {
        'status': {
            'experiment': 1,
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
            'cycle': 1
        },
        'resources': {
            'activation-date': datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
            'physical-cpu-amount': psutil.cpu_count(logical=False),
            'total-cpu-amount': psutil.cpu_count(logical=True),
            'min-cpu-frequency-mhz': psutil.cpu_freq().min,
            'max-cpu-frequency-mhz': psutil.cpu_freq().max,
            'total-ram-amount-bytes': psutil.virtual_memory().total,
            'available-ram-amount-bytes': psutil.virtual_memory().free,
            'total-disk-amount-bytes': psutil.disk_usage('.').total,
            'available-disk-amount-bytes': psutil.disk_usage('.').free
        },
        'templates/model-parameters': {
            'seed': 0,
            'used-columns': [],
            'input-size': 0,
            'target-column': '',
            'scaled-columns': [],
            'learning-rate': 0.0,
            'sample-rate': 0.0,
            'optimizer': '',
            'epochs': 0
        },
        'templates/central-parameters': {
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
        },
        'templates/worker-parameters': {
            'sample-pool': 0,
            'data-augmentation': {
                'active': False,
                'sample-pool': 0,
                '1-0-ratio': 0.0
            },
            'eval-ratio': 0.0,
            'train-ratio': 0.0
        }
        
    }
    central_bucket = 'central'
    for key in templates.keys():
        given_object_path = 'experiments/' + str(key)
        
        object_exists = check_object(
            logger = logger,
            minio_client = minio_client,
            bucket_name = central_bucket,
            object_path = given_object_path
        )
        if not object_exists:
            create_or_update_object(
                logger = logger,
                minio_client = minio_client,
                bucket_name = central_bucket,
                object_path = given_object_path,
                data = templates[key],
                metadata = {}
            )