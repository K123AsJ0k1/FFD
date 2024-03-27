import os 
import json
import requests
import psutil
import time
import pandas as pd
from functions.general import format_metadata_dict, encode_metadata_lists_to_strings
from functions.management.storage import store_metrics_and_resources
from functions.platforms.minio import get_object_data_and_metadata, create_or_update_object

# Refactored
def send_context_to_workers(
    file_lock: any,
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any
) -> bool:
    this_process = psutil.Process(os.getpid())
    func_mem_start = psutil.virtual_memory().used 
    func_disk_start = psutil.disk_usage('.').used
    func_cpu_start = this_process.cpu_percent(interval=0.2)
    func_time_start = time.time()

    experiments_folder = 'experiments'
    central_bucket = 'central'
    central_status_path = experiments_folder + '/status'
    central_status_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path
    )
    central_status = central_status_object['data']

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

    experiment_folder_path = experiments_folder + '/' + str(central_status['experiment'])
    cycle_folder_path = experiment_folder_path + '/' + str(central_status['cycle'])
    workers_status_path = cycle_folder_path + '/' + 'workers'
    workers_status_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = workers_status_path
    )
    if workers_status_object is None:
        return False
    workers_status = workers_status_object['data']
    
    parameters_folder_path = experiment_folder_path + '/parameters'
    central_parameters_path = parameters_folder_path + '/central'
    model_parameters_path = parameters_folder_path + '/model'
    worker_parameters_path = parameters_folder_path + '/worker'

    central_parameters_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_parameters_path
    )
    central_parameters = central_parameters_object['data']

    model_parameters_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = model_parameters_path
    )
    model_parameters = model_parameters_object['data']

    worker_parameters_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = worker_parameters_path
    )
    worker_parameters = worker_parameters_object['data']

    global_model_path = cycle_folder_path + '/global-model'

    global_model_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = global_model_path
    )
    global_model = global_model_object['data']

    formatted_global_model = {
        'weights': global_model['linear.weight'].numpy().tolist(),
        'bias': global_model['linear.bias'].numpy().tolist()
    }
    # Refactor to have failure fixing 
    success = False
    for i in range(0,10):
        if success:
            break

        payload_status = {}
        data_folder_path = cycle_folder_path + '/data'
        
        for worker_key in workers_status.keys():
            this_process = psutil.Process(os.getpid())
            net_mem_start = psutil.virtual_memory().used 
            net_disk_start = psutil.disk_usage('.').used
            net_cpu_start = this_process.cpu_percent(interval=0.2)
            net_time_start = time.time()
            
            worker_status = workers_status[worker_key]
            if worker_status['stored']:
                continue

            context = None
            if not central_status['complete']:
                worker_pool_path = data_folder_path + '/' + worker_key
                worker_pool_object = get_object_data_and_metadata(
                    logger = logger,
                    minio_client = minio_client,
                    bucket_name = central_bucket,
                    object_path = worker_pool_path
                )
                worker_data_list = worker_pool_object['data']
                worker_data_columns = format_metadata_dict(worker_pool_object['metadata'])['columns']
                
                info = {
                    'worker-id': worker_key,
                    'experiment': central_status['experiment'],
                    'cycle': central_status['cycle'],
                    'model': model_parameters,
                    'worker': worker_parameters
                }
                
                context = {
                    'info': info,
                    'global-model': formatted_global_model,
                    'worker-data-list': worker_data_list,
                    'worker-data-columns': worker_data_columns
                }
            else:
                info = {
                    'worker-id': worker_key,
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
                net_cpu_end = this_process.cpu_percent(interval=0.2)
                net_mem_end = psutil.virtual_memory().used 
                net_disk_end = psutil.disk_usage('.').used

                net_time_diff = (net_time_end - net_time_start) 
                net_cpu_diff = net_cpu_end - net_cpu_start 
                net_mem_diff = (net_mem_end - net_mem_start) 
                net_disk_diff = (net_disk_end - net_disk_start) 

                resource_metrics = {
                    'name': 'sending-context-to-worker-' + str(worker_key),
                    'status-code': response.status_code,
                    'processing-time-seconds': net_time_diff,
                    'elapsed-time-seconds': response.elapsed.total_seconds(),
                    'cpu-percentage': net_cpu_diff,
                    'ram-bytes': net_mem_diff,
                    'disk-bytes': net_disk_diff
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
        times_path = experiment_folder_path + '/times'

        times_object = get_object_data_and_metadata(
            logger = logger,
            minio_client = minio_client,
            bucket_name = central_bucket,
            object_path = times_path
        )
        times = times_object['data']

        experiment_start = times['experiment-time-start']
        experiment_end = time.time()
        experiment_total = experiment_end - experiment_start
        times['experiment-time-end'] = experiment_end
        times['experiment-total-seconds'] = experiment_total

        create_or_update_object(
            logger = logger,
            minio_client = minio_client,
            bucket_name = central_bucket,
            object_path = times_path,
            data = times,
            metadata = {}
        )
       
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path,
        data = central_status,
        metadata = {}
    )

    os.environ['STATUS'] = 'context sent to workers'
    logger.info('Context sent to workers')

    func_time_end = time.time()
    func_cpu_end = this_process.cpu_percent(interval=0.2)
    func_mem_end = psutil.virtual_memory().used 
    func_disk_end = psutil.disk_usage('.').used

    func_time_diff = (func_time_end - func_time_start) 
    func_cpu_diff = func_cpu_end - func_cpu_start 
    func_mem_diff = (func_mem_end - func_mem_start)
    func_disk_diff = (func_disk_end - func_disk_start) 

    resource_metrics = {
        'name': 'send-context-to-workers',
        'time-seconds': func_time_diff,
        'cpu-percentage': func_cpu_diff,
        'ram-bytes': func_mem_diff,
        'disk-bytes': func_disk_diff
    }

    store_metrics_and_resources(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        type = 'resources',
        area = 'function',
        metrics = resource_metrics
    )
    
    return True