from functions.platforms.minio import create_or_update_object, check_object
import psutil
from datetime import datetime
from prometheus_client import Gauge

# Refactored and works
def initilize_minio(
    logger: any,
    minio_client: any
):
    templates = {
        'status': {
            'experiment': 1,
            'experiment-id': '',
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
def initilize_prometheus_gauges(
    prometheus_registry: any,
    prometheus_metrics: any,
):
    # Global model metrics
    #metric_name = 'central_global'
    prometheus_metrics['central-global'] = Gauge(
        name = 'C_M_M',
        documentation = 'Central global metrics',
        labelnames = ['date', 'experiment','cycle','metric'],
        registry = prometheus_registry
    )
    # Global metric names
    prometheus_metrics['central-global-names'] = {
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
    # Global resource metrics
    #metric_name = 'central-resources'
    prometheus_metrics['central-resources'] = Gauge(
        name = 'C_R_M',
        documentation = 'Central resource metrics',
        labelnames = ['date', 'experiment','cycle','area','name','metric'],
        registry = prometheus_registry
    )
    prometheus_metrics['central-resources-names'] = {
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