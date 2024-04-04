import psutil
import os

from datetime import datetime

from functions.management.objects import get_experiments_objects, set_experiments_objects

from functions.platforms.prometheus import worker_local_gauge, worker_resources_gauge, worker_time_gauge
# Created and works
def initilize_envs(
    file_lock: any,
    logger: any,
    minio_client: any
):
    worker_status, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = ''
    )
    if not worker_status is None:
        os.environ['EXP_NAME'] = str(worker_status['experiment-name'])
        os.environ['EXP'] = str(worker_status['experiment'])
        os.environ['CYCLE'] = str(worker_status['cycle'])
    else:
        os.environ['CYCLE'] = str(1)
        os.environ['EXP'] = str(1)
        os.environ['EXP_NAME'] = ''
# Created and works
def initilize_minio(
    file_lock: any,
    logger: any,
    minio_client: any
):
    worker_id = os.environ.get('WORKER_ID')
    central_address = os.environ.get('CENTRAL_ADDRESS')
    central_port = os.environ.get('CENTRAL_PORT')
    worker_port = os.environ.get('WORKER_PORT')
    templates = {
        'status': {
            'worker-id': worker_id,
            'network-id': 0,
            'central-address': central_address,
            'central-port': central_port,
            'worker-address': '',
            'worker-port': worker_port,
            'status': 'waiting',
            'experiment-name': '',
            'experiment':1,
            'experiment-id': '',
            'stored': False,
            'preprocessed': False,
            'trained': False,
            'updated': False,
            'complete': False,
            'train-amount': 0,
            'test-amount':0,
            'eval-amount': 0,
            'cycle': 1
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
    local_metrics, local_metrics_names = worker_local_gauge(
        prometheus_registry = prometheus_registry
    ) 
    prometheus_metrics['local'] = local_metrics
    prometheus_metrics['local-name'] = local_metrics_names
    resource_metrics, resource_metrics_names = worker_resources_gauge(
        prometheus_registry = prometheus_registry
    ) 
    prometheus_metrics['resource'] = resource_metrics
    prometheus_metrics['resource-name'] = resource_metrics_names
    time_metrics, time_metrics_names = worker_time_gauge(
        prometheus_registry = prometheus_registry
    ) 
    prometheus_metrics['time'] = time_metrics
    prometheus_metrics['time-name'] = time_metrics_names