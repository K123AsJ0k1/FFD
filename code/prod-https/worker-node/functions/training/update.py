import os
import json
import requests
import psutil
import time

from functions.management.storage import store_metrics_resources_and_times
from functions.platforms.minio import get_object_data_and_metadata, create_or_update_object
from functions.general import get_experiments_objects, set_experiments_objects
# Refactored
def send_info_to_central(
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any
) -> any:
    time_start = time.time()

    worker_status, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = ''
    )

    if worker_status is None:
        return False, None

    if worker_status['central-address'] == '':
        return False, None
    
    if worker_status['trained'] and not worker_status['updated']:
        return False, None
    
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
                    worker_status['network-id'] = sent_payload['network-id']
                    worker_status['worker-address'] = sent_payload['worker-address']
                    worker_status['experiment-name'] = sent_payload['experiment-name']
                    worker_status['experiment'] = sent_payload['experiment']
                    worker_status['cycle'] = sent_payload['cycle']
                    os.environ['EXP_NAME'] = str(sent_payload['experiment-name'])
                    os.environ['EXP'] = str(sent_payload['experiment'])
                    os.environ['CYCLE'] = str(sent_payload['cycle'])
                # Worker id is known, but address has changed
                if message == 'rerouted':
                    worker_status['worker-address'] = sent_payload['worker-address']
                
                set_experiments_objects(
                    logger = logger,
                    minio_client = minio_client,
                    object = 'status',
                    replacer = '',
                    overwrite = True,
                    object_data = worker_status,
                    object_metadata = {}
                )

        time_end = time.time()
        time_diff = (time_end - time_start) 
        resource_metrics = {
            'name': 'sending-info-to-central',
            'status-code': response.status_code,
            'processing-time-seconds': time_diff,
            'elapsed-time-seconds': response.elapsed.total_seconds(),
            'action-time-start': time_start,
            'action-time-end': time_end,
            'action-total-seconds': round(time_diff,5)
        }
        
        store_metrics_resources_and_times(
            logger = logger,
            minio_client = minio_client,
            prometheus_registry = prometheus_registry,
            prometheus_metrics = prometheus_metrics,
            type = 'times',
            area = 'network',
            metrics = resource_metrics
        )
        
        return True, response.status_code
    except Exception as e:
        logger.error('Sending info to central error:' +  str(e)) 
        return False, None
# Refactored
def send_update_to_central(
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any
) -> bool:  
    time_start = time.time()

    worker_status, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = ''
    )

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
 
    local_model, local_model_details = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'model',
        replacer = 'local-model'
    )

    formatted_local_model = {
      'weights': local_model['linear.weight'].numpy().tolist(),
      'bias': local_model['linear.bias'].numpy().tolist()
    }

    update = {
        'worker-id': str(worker_status['worker-id']),
        'experiment-name': str(worker_status['experiment-name']),
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

        if status_code == 200 and (message == 'stored' or message == 'late'):
            experiment_times, _ = get_experiments_objects(
                logger = logger,
                minio_client = minio_client,
                object = 'experiment-times',
                replacer = ''
            )

            cycle_start = experiment_times[str(worker_status['cycle'])]['cycle-time-start']
            cycle_end = time.time()
            cycle_total = cycle_end-cycle_start
            experiment_times[str(worker_status['cycle'])]['cycle-time-end'] = cycle_end
            experiment_times[str(worker_status['cycle'])]['cycle-total-seconds'] = cycle_total
            
            set_experiments_objects(
                logger = logger,
                minio_client = minio_client,
                object = 'experiment-times',
                replacer = '',
                overwrite = True,
                object_data = experiment_times,
                object_metadata = {}
            )
            
            worker_status['updated'] = True
            worker_status['stored'] = False
            set_experiments_objects(
                logger = logger,
                minio_client = minio_client,
                object = 'status',
                replacer = '',
                overwrite = True,
                object_data = worker_status,
                object_metadata = {}
            )

            os.environ['STATUS'] = 'update sent to central'
            logger.info('Update sent to central')

            time_end = time.time()
            time_diff = (time_end - time_start) 
        
            resource_metrics = {
                'name': 'sending-update-to-central',
                'status-code': response.status_code,
                'processing-time-seconds': time_diff,
                'elapsed-time-seconds': response.elapsed.total_seconds(),
                'action-time-start': time_start,
                'action-time-end': time_end,
                'action-total-seconds': round(time_diff,5)
            }

            store_metrics_resources_and_times(
                logger = logger,
                minio_client = minio_client,
                prometheus_registry = prometheus_registry,
                prometheus_metrics = prometheus_metrics,
                type = 'times',
                area = 'network',
                metrics = resource_metrics
            )
            success = True
            continue 
        
        os.environ['STATUS'] = 'update not sent to central'
        logger.info('Update not sent to central')

        time_end = time.time()
        time_diff = (time_end - time_start) 
        resource_metrics = {
            'name': 'sending-update-to-central',
            'status-code': response.status_code,
            'processing-time-seconds': time_diff,
            'elapsed-time-seconds': response.elapsed.total_seconds(),
            'action-time-start': time_start,
            'action-time-end': time_end,
            'action-total-seconds': round(time_diff,5)
        }

        store_metrics_resources_and_times(
            logger = logger,
            minio_client = minio_client,
            prometheus_registry = prometheus_registry,
            prometheus_metrics = prometheus_metrics,
            type = 'times',
            area = 'network',
            metrics = resource_metrics
        )
    return True