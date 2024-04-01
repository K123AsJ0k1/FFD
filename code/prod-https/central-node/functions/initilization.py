import psutil
import os

from datetime import datetime
from functions.general import set_experiments_objects, get_experiments_objects
from functions.platforms.prometheus import central_global_gauge, central_time_gauge, central_resource_gauge
# Created and works
def initilize_envs(
    file_lock: any,
    logger: any,
    minio_client: any
):
    central_status, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = ''
    )
    if not central_status is None:
        os.environ['EXP_NAME'] = str(central_status['experiment-name'])
        os.environ['EXP'] = str(central_status['experiment'])
        os.environ['CYCLE'] = str(central_status['cycle'])
    else:
        os.environ['CYCLE'] = str(1)
        os.environ['EXP'] = str(1)
        os.environ['EXP_NAME'] = ''
# Refactored and works        
def initilize_minio(
    file_lock: any,
    logger: any,
    minio_client: any
):  
    templates = {
        'status': {
            'experiment-name': '',
            'experiment': 1,
            'experiment-id': '',
            'start': False,
            'data-split': False,
            'preprocessed': False,
            'trained': False,
            'worker-split': False,
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
            'run-id': 0
        },
        'specifications': {
            'activation-date': datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
            'host-kernel-version': os.uname().release,
            'host-system-name': os.uname().sysname,
            'host-node-name': os.uname().nodename,
            'host-machine': os.uname().machine,
            'physical-cpu-amount': psutil.cpu_count(logical=False),
            'total-cpu-amount': psutil.cpu_count(logical=True),
            'min-cpu-frequency-mhz': psutil.cpu_freq().min,
            'max-cpu-frequency-mhz': psutil.cpu_freq().max,
            'total-ram-amount-bytes': psutil.virtual_memory().total,
            'available-ram-amount-bytes': psutil.virtual_memory().free,
            'total-disk-amount-bytes': psutil.disk_usage('.').total,
            'available-disk-amount-bytes': psutil.disk_usage('.').free
        },
        'central-template': {
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
        'model-template': {
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
        'worker-template': {
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

    for key in templates.keys():
        set_experiments_objects(
            file_lock = file_lock,
            logger = logger,
            minio_client = minio_client,
            object = key,
            replacer = '',
            overwrite = False,
            object_data = templates[key],
            object_metadata = {} 
        )    
# Created and works       
def initilize_prometheus_gauges(
    prometheus_registry: any,
    prometheus_metrics: any,
):
    global_metrics, global_metrics_names = central_global_gauge(
        prometheus_registry = prometheus_registry
    ) 
    prometheus_metrics['global'] = global_metrics
    prometheus_metrics['global-name'] = global_metrics_names
    resource_metrics, resource_metrics_names = central_resource_gauge(
        prometheus_registry = prometheus_registry
    ) 
    prometheus_metrics['resource'] = resource_metrics
    prometheus_metrics['resource-name'] = resource_metrics_names
    time_metrics, time_metrics_names = central_time_gauge(
        prometheus_registry = prometheus_registry
    ) 
    prometheus_metrics['time'] = time_metrics
    prometheus_metrics['time-name'] = time_metrics_names