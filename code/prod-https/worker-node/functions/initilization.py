from functions.platforms.minio import get_object_list, create_or_update_object, check_object
import psutil
from datetime import datetime
import os
from prometheus_client import Gauge

# Created and works
def initilize_minio(
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
            'experiment':1,
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

    workers_bucket = 'workers'
    for key in templates.keys():
        given_object_path = worker_id + '/experiments/' + str(key)
        
        object_exists = check_object(
            logger = logger,
            minio_client = minio_client,
            bucket_name = workers_bucket,
            object_path = given_object_path
        )
        if not object_exists:
            create_or_update_object(
                logger = logger,
                minio_client = minio_client,
                bucket_name = workers_bucket,
                object_path = given_object_path,
                data = templates[key],
                metadata = {}
            )

def initilize_prometheus_gauges(
    prometheus_registry: any,
    prometheus_metrics: any,
):
    # Worker model metrics
    #metric_name = 'central_global'
    prometheus_metrics['worker-local'] = Gauge(
        name = 'W_M_M',
        documentation = 'Worker local metrics',
        labelnames = ['date','woid','neid','cead','woad','experiment','cycle','metric'],
        registry = prometheus_registry
    )
    # Worker metric names
    prometheus_metrics['worker-local-names'] = {
        'train-amount': 'TrAm',
        'test-amount': 'TeAm',
        'eval-amount': 'EvAm',
        'true-positives': 'TrPo',
        'false-positives': 'FaPo',
        'true-negatives': 'TrNe',
        'false-negatives': 'FaNe',
        'recall': 'ReMe',
        'selectivity': 'SeMe',
        'precision': 'PrMe',
        'miss-rate': 'MiRaMe',
        'fall-out': 'FaOuMe',
        'balanced-accuracy': 'BaAcMe',
        'accuracy': 'AcMe'
    }
    # Worker resource metrics
    #metric_name = 'central-resources'
    prometheus_metrics['worker-resources'] = Gauge(
        name = 'W_R_M',
        documentation = 'Central resource metrics',
        labelnames = ['date','woid','neid','cead','woad','experiment','cycle','area','name','metric'],
        registry = prometheus_registry
    )
    prometheus_metrics['worker-resources-names'] = {
        'physical-cpu-amount': 'PyCPUAm',
        'total-cpu-amount': 'ToCPUAm',
        'min-cpu-frequency-mhz': 'MinCPUFrMhz',
        'max-cpu-frequency-mhz': 'MaxCPUFrMhz',
        'total-ram-amount-bytes': 'ToRAMAmBy',
        'available-ram-amount-bytes': 'AvRAMAmByte',
        'total-disk-amount-bytes': 'ToDiAmBy',
        'available-disk-amount-bytes': 'ToDiAmByte',
        'experiment-date': 'ExDa',
        'experiment-time-start':'ExTiSt',
        'experiment-time-end':'ExTiEn',
        'experiment-total-seconds': 'ExToSec',
        'cycle-time-start': 'CyTiSt',
        'cycle-time-end': 'CyTiEn',
        'cycle-total-seconds': 'CyToSec',
        'time-seconds': 'TiSec',
        'processing-time-seconds': 'PrTiSec',
        'elapsed-time-seconds': 'ElTiSec',
        'cpu-percentage': 'CPUPerc',
        'ram-bytes': 'RAMByte',
        'disk-bytes': 'DiByte',
        'epochs': 'Epo',
        'batches': 'Bat',
        'average-batch-size': 'AvBatSi'
    }