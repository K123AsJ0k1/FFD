import os
import json
import requests
import psutil
import time
from datetime import datetime

from functions.management.storage import store_metrics_and_resources
from functions.platforms.minio import get_object_data_and_metadata, create_or_update_object
# Refactored and works
def send_info_to_central(
    file_lock: any,
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any
) -> any:
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    workers_bucket = 'workers'
    worker_experiments_folder = os.environ.get('WORKER_ID') + '/experiments'
    worker_status_path = worker_experiments_folder + '/status'
    worker_status_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = workers_bucket,
        object_path = worker_status_path
    )
    worker_status = worker_status_object['data']

    if worker_status is None:
        return False

    if worker_status['central-address'] == '':
        return False
    
    try:
        info = {
            'status': worker_status
        }
        central_url = 'http://' + worker_status['central-address'] + ':' + worker_status['central-port'] + '/status'
        payload = json.dumps(info) 
        
        response = requests.post(
            url = central_url,
            json = payload,
            headers = {
               'Content-type':'application/json', 
               'Accept':'application/json'
            }
        )
        
        if response.status_code == 200:
            sent_payload = json.loads(response.text)
            message = sent_payload['message']
            
            if message == 'registered' or message == 'rerouted':
                # Worker is either new or new experiment has been started
                if message == 'registered':
                    worker_status['experiment'] = sent_payload['experiment']
                    worker_status['network-id'] = sent_payload['network-id']
                    worker_status['worker-address'] = sent_payload['worker-address']
                    worker_status['cycle'] = sent_payload['cycle']
                # Worker id is known, but address has changed
                if message == 'rerouted':
                    worker_status['worker-address'] = sent_payload['worker-address']

                create_or_update_object(
                    logger = logger,
                    minio_client = minio_client,
                    bucket_name = workers_bucket,
                    object_path = worker_status_path,
                    data = worker_status,
                    metadata = {}
                )

        time_end = time.time()
        cpu_end = this_process.cpu_percent(interval=0.2)
        mem_end = psutil.virtual_memory().used 
        disk_end = psutil.disk_usage('.').used

        time_diff = (time_end - time_start) 
        cpu_diff = cpu_end - cpu_start 
        mem_diff = (mem_end - mem_start) 
        disk_diff = (disk_end - disk_start) 

        resource_metrics = {
            'name': 'sending-info-to-central',
            'status-code': response.status_code,
            'processing-time-seconds': time_diff,
            'elapsed-time-seconds': response.elapsed.total_seconds(),
            'cpu-percentage': cpu_diff,
            'ram-bytes': mem_diff,
            'disk-bytes': disk_diff
        }
        
        store_metrics_and_resources(
            file_lock = file_lock,
            logger = logger,
            minio_client = minio_client,
            prometheus_registry = prometheus_registry,
            prometheus_metrics = prometheus_metrics,
            type = 'resources',
            area = 'network',
            metrics = resource_metrics
        )
        
        return True, response.status_code
    except Exception as e:
        logger.error('Sending info to central error:' +  str(e)) 
        return False, None
# Refactored
def send_update_to_central(
    file_lock: any,
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any
) -> bool:  
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    workers_bucket = 'workers'
    worker_experiments_folder = os.environ.get('WORKER_ID') + '/experiments'
    worker_status_path = worker_experiments_folder + '/status'
    worker_status_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = workers_bucket,
        object_path = worker_status_path
    )
    worker_status = worker_status_object['data']

    if worker_status is None:
        return False
    
    if not worker_status['stored'] or not worker_status['preprocessed'] or not worker_status['trained']:
        return False

    if worker_status['complete']:
        return False

    if worker_status['updated']:
        return False

    os.environ['STATUS'] = 'sending update to central'
    logger.info('Sending update to central')

    experiment_folder_path = worker_experiments_folder + '/' + str(worker_status['experiment'])
    cycle_folder_path = experiment_folder_path + '/' + str(worker_status['cycle'])

    local_model_path = cycle_folder_path + '/local-model'
    local_model_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = workers_bucket,
        object_path = local_model_path
    )
    local_model = local_model_object['data']

    formatted_local_model = {
      'weights': local_model['linear.weight'].numpy().tolist(),
      'bias': local_model['linear.bias'].numpy().tolist()
    }

    update = {
        'worker-id': str(worker_status['worker-id']),
        'experiment': str(worker_status['experiment']),
        'cycle': str(worker_status['cycle']),
        'local-model': formatted_local_model
    }

    payload = json.dumps(update)
    central_url = 'http://' + worker_status['central-address'] + ':' + worker_status['central-port'] + '/update'
    success = False
    for i in range(0,50):
        if success:
            break
        
        status_code = None
        message = None
        try:
            central_url = 'http://' + worker_status['central-address'] + ':' + worker_status['central-port'] + '/update'
            response = requests.post(
                url = central_url, 
                json = payload,
                headers = {
                    'Content-type':'application/json', 
                    'Accept':'application/json'
                }
            )
            status_code = response.status_code
            message = json.loads(response.text)['message']
        except Exception as e:
            logger.error('Status sending error:' + str(e))

        if status_code == 200 and message == 'stored':
            times_path = experiment_folder_path + '/times'
            times_object = get_object_data_and_metadata(
                logger = logger,
                minio_client = minio_client,
                bucket_name = workers_bucket,
                object_path = times_path
            )
            times = times_object['data']

            cycle_start = times[str(worker_status['cycle'])]['cycle-time-start']
            cycle_end = time.time()
            cycle_total = cycle_end-cycle_start
            times[str(worker_status['cycle'])]['cycle-time-end'] = cycle_end
            times[str(worker_status['cycle'])]['cycle-total-seconds'] = cycle_total

            create_or_update_object(
                logger = logger,
                minio_client = minio_client,
                bucket_name = workers_bucket,
                object_path = times_path,
                data = times,
                metadata = {}
            )
            
            worker_status['updated'] = True
            worker_status['stored'] = False
            create_or_update_object(
                logger = logger,
                minio_client = minio_client,
                bucket_name = workers_bucket,
                object_path = worker_status_path,
                data = worker_status,
                metadata = {}
            )

            os.environ['STATUS'] = 'update sent to central'
            logger.info('Update sent to central')

            time_end = time.time()
            cpu_end = this_process.cpu_percent(interval=0.2)
            mem_end = psutil.virtual_memory().used 
            disk_end = psutil.disk_usage('.').used

            time_diff = (time_end - time_start) 
            cpu_diff = cpu_end - cpu_start 
            mem_diff = (mem_end - mem_start) 
            disk_diff = (disk_end - disk_start) 

            resource_metrics = {
                'name': 'sending-update-to-central',
                'status-code': response.status_code,
                'processing-time-seconds': time_diff,
                'elapsed-time-seconds': response.elapsed.total_seconds(),
                'cpu-percentage': cpu_diff,
                'ram-bytes': mem_diff,
                'disk-bytes': disk_diff
            }

            store_metrics_and_resources(
                file_lock = file_lock,
                logger = logger,
                minio_client = minio_client,
                prometheus_registry = prometheus_registry,
                prometheus_metrics = prometheus_metrics,
                type = 'resources',
                area = 'network',
                metrics = resource_metrics
            )
            success = True
            continue 
        
        os.environ['STATUS'] = 'update not sent to central'
        logger.info('Update not sent to central')

        time_end = time.time()
        cpu_end = this_process.cpu_percent(interval=0.2)
        mem_end = psutil.virtual_memory().used 
        disk_end = psutil.disk_usage('.').used

        time_diff = (time_end - time_start) 
        cpu_diff = cpu_end - cpu_start 
        mem_diff = (mem_end - mem_start) 
        disk_diff = (disk_end - disk_start)

        resource_metrics = {
            'name': 'sending-update-to-central',
            'status-code': response.status_code,
            'processing-time-seconds': time_diff,
            'elapsed-time-seconds': response.elapsed.total_seconds(),
            'cpu-percentage': cpu_diff,
            'ram-bytes': mem_diff,
            'disk-bytes': disk_diff
        }

        store_metrics_and_resources(
            file_lock = file_lock,
            logger = logger,
            minio_client = minio_client,
            prometheus_registry = prometheus_registry,
            prometheus_metrics = prometheus_metrics,
            type = 'resources',
            area = 'network',
            metrics = resource_metrics
        )
    return True