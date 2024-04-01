import os 
import json
import requests
import psutil
import time
from functions.general import get_experiments_objects, set_experiments_objects
from functions.management.storage import store_metrics_resources_and_times
from functions.platforms.minio import get_object_data_and_metadata, create_or_update_object
from functions.platforms.mlflow import start_run

# Refactored
def send_context_to_workers(
    logger: any,
    minio_client: any,
    mlflow_client: any,
    prometheus_registry: any,
    prometheus_metrics: any
) -> bool:
    time_start = time.time()

    central_status, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = ''
    )

    if central_status is None:
        return False

    if not central_status['start']:
        return False

    if not central_status['worker-split']:
        return False
    
    if not central_status['trained']:
        return False
    
    if central_status['sent']:
        return False
    
    os.environ['STATUS'] = 'sending context to workers'
    logger.info('Sending context to workers')

    workers_status, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'workers',
        replacer = ''
    )

    central_parameters, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'parameters',
        replacer = 'central'
    )

    model_parameters, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'parameters',
        replacer = 'model'
    )

    worker_parameters, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'parameters',
        replacer = 'worker'
    )
    
    global_model, _ = get_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'global-model',
        replacer = ''
    )
    
    formatted_global_model = {
        'weights': global_model['linear.weight'].numpy().tolist(),
        'bias': global_model['linear.bias'].numpy().tolist()
    }

    if not central_status['complete']:
        run_data = start_run(
            logger = logger,
            mlflow_client = mlflow_client,
            experiment_id = central_status['experiment-id'],
            tags = {},
            name = 'federated-training' 
        )
        central_status['run-id'] = run_data['id']

    # Refactor to have failure fixing 
    success = False
    for i in range(0,10):
        if success:
            break

        payload_status = {}
        for worker_key in workers_status.keys():
            net_time_start = time.time()
            
            worker_status = workers_status[worker_key]
            if worker_status['stored']:
                continue

            context = None
            if not central_status['complete']:
                worker_data, worker_data_details = get_experiments_objects(
                    logger = logger,
                    minio_client = minio_client,
                    object = 'data-worker',
                    replacer = worker_key
                )
               
                info = {
                    'worker-id': worker_key,
                    'experiment-name': central_status['experiment-name'],
                    'experiment': central_status['experiment'],
                    'cycle': central_status['cycle'],
                    'model': model_parameters,
                    'worker': worker_parameters
                }
                
                context = {
                    'info': info,
                    'global-model': formatted_global_model,
                    'worker-data-list': worker_data,
                    'worker-data-columns': worker_data_details['header']
                }
            else:
                info = {
                    'worker-id': worker_key,
                    'experiment-name': central_status['experiment-name'],
                    'experiment': central_status['experiment'],
                    'cycle': central_status['cycle'],
                    'model': None,
                    'worker': None
                }

                context = {
                    'info': info,
                    'global-model': formatted_global_model,
                    'worker-data-list': None,
                    'worker-data-columns': None
                }
        
            json_payload = json.dumps(context) 
            try:
                worker_url = 'http://' + worker_status['worker-address'] + ':' + worker_status['worker-port'] + '/context'
                response = requests.post(
                    url = worker_url, 
                    json = json_payload,
                    headers = {
                        'Content-type':'application/json', 
                        'Accept':'application/json'
                    }
                )

                sent_message = json.loads(response.text)['message']
                payload_status[worker_key] = {
                    'response': response.status_code,
                    'message': sent_message
                }

                net_time_end = time.time()
                net_time_diff = (net_time_end - net_time_start) 
                time_end = time.time()
                time_diff = (time_end - time_start) 
                
                resource_metrics = {
                    'name': 'sending-context-to-worker-' + str(worker_key),
                    'status-code': response.status_code,
                    'processing-time-seconds': net_time_diff,
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
            except Exception as e:
                logger.error('Context sending error:' + str(e))
        
        successes = 0
        for worker_key in payload_status.keys():
            worker_data = payload_status[worker_key]
            if worker_data['response'] == 200 :
                if worker_data['message'] == 'stored' or worker_data['message'] == 'ongoing':
                    successes = successes + 1
                continue
            successes = successes + 1 
        # Could be reconsidered
        if central_parameters['min-update-amount'] <= successes:
            central_status['sent'] = True
            success = True
            if not central_status['complete']:
                os.environ['STATUS'] = 'waiting updates'
            else: 
                os.environ['STATUS'] = 'training complete'

    if central_status['complete']:
        experiment_times, _ = get_experiments_objects(
            logger = logger,
            minio_client = minio_client,
            object = 'experiment-times',
            replacer = ''
        )
        
        experiment_start = experiment_times['experiment-time-start']
        experiment_end = time.time()
        experiment_total = experiment_end - experiment_start
        experiment_times['experiment-time-end'] = experiment_end
        experiment_times['experiment-total-seconds'] = experiment_total

        set_experiments_objects(
            logger = logger,
            minio_client = minio_client,
            object = 'experiment-times',
            replacer = '',
            overwrite = True,
            object_data = experiment_times,
            object_metadata = {}
        )

    set_experiments_objects(
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = '',
        overwrite = True,
        object_data = central_status,
        object_metadata = {}
    )

    os.environ['STATUS'] = 'context sent to workers'
    logger.info('Context sent to workers')

    time_end = time.time()
    time_diff = (time_end - time_start) 
    resource_metrics = {
        'name': 'send-context-to-workers',
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
        area = 'function',
        metrics = resource_metrics
    )
    
    return True