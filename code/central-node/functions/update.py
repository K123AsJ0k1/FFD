import os 
import json
import requests
import psutil
import time

from functions.general import get_current_experiment_number, get_wanted_model, get_file_data, get_files
from functions.storage import store_metrics_and_resources, store_file_data

# Refactored and works 
def send_context_to_workers(
    file_lock: any,
    logger: any
) -> bool:
    this_process = psutil.Process(os.getpid())
    func_mem_start = psutil.virtual_memory().used 
    func_disk_start = psutil.disk_usage('.').used
    func_cpu_start = this_process.cpu_percent(interval=0.2)
    func_time_start = time.time()

    current_experiment_number = get_current_experiment_number()
    status_folder_path = 'status/experiment_' + str(current_experiment_number)
    central_status_path = status_folder_path + '/central.txt'

    central_status = get_file_data(
        file_lock = file_lock,
        file_path = central_status_path
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

    workers_status_path = status_folder_path + '/workers.txt'
    workers_status = get_file_data(
        file_lock = file_lock,
        file_path = workers_status_path
    )

    parameters_folder_path = 'parameters/experiment_' + str(current_experiment_number)
    central_parameters_path = parameters_folder_path + '/central.txt'
    model_parameters_path = parameters_folder_path + '/model.txt'
    worker_parameters_path = parameters_folder_path + '/worker.txt'

    central_parameters = get_file_data(
        file_lock = file_lock,
        file_path = central_parameters_path
    )

    if central_parameters is None:
        return False

    model_parameters = get_file_data(
        file_lock = file_lock,
        file_path = model_parameters_path
    )

    if model_parameters is None:
        return False

    worker_parameters = get_file_data(
        file_lock = file_lock,
        file_path = worker_parameters_path
    )

    if worker_parameters is None:
        return False
    
    global_model = get_wanted_model(
        file_lock = file_lock,
        experiment = current_experiment_number,
        subject = 'global',
        cycle = central_status['cycle']-1
    )

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
        data_folder_path = 'data/experiment_' + str(current_experiment_number)
        data_files = get_files(data_folder_path)
        for worker_key in workers_status.keys():
            this_process = psutil.Process(os.getpid())
            net_mem_start = psutil.virtual_memory().used 
            net_disk_start = psutil.disk_usage('.').used
            net_cpu_start = this_process.cpu_percent(interval=0.2)
            net_time_start = time.time()
            
            worker_metadata = workers_status[worker_key]
            if worker_metadata['stored']:
                continue

            worker_url = worker_metadata['worker-address'] + '/context'
            payload = None
            if not central_status['complete']:
                data_path = ''
                for data_file in data_files:
                    first_split = data_file.split('.')
                    second_split = first_split[0].split('_')
                    if second_split[0] == 'worker':
                        if second_split[1] == worker_key and second_split[2] == str(central_status['cycle']):
                            data_path = data_folder_path + '/' + data_file
                
                worker_df = get_file_data(
                    file_lock = file_lock,
                    file_path = data_path
                )
                worker_data_list = worker_df.values.tolist()
                worker_data_columns = worker_df.columns.tolist()

                parameters = {
                    'id': worker_key,
                    'worker-address': worker_metadata['worker-address'],
                    'cycle': central_status['cycle'],
                    'model': model_parameters,
                    'worker': worker_parameters
                }
                
                payload = {
                    'parameters': parameters,
                    'global-model': formatted_global_model,
                    'worker-data-list': worker_data_list,
                    'worker-data-columns': worker_data_columns
                }
            else:
                parameters = {
                    'id': worker_key,
                    'worker-address': worker_metadata['worker-address'],
                    'cycle': central_status['cycle'],
                    'model': None,
                    'worker': None
                }

                payload = {
                    'parameters': parameters,
                    'global-model': formatted_global_model,
                    'worker-data-list': None,
                    'worker-data-columns': None
                }
        
            json_payload = json.dumps(payload) 
            try:
                response = requests.post(
                    url = worker_url, 
                    json = json_payload,
                    headers = {
                        'Content-type':'application/json', 
                        'Accept':'application/json'
                    }
                )

                sent_message = json.loads(response.text)
                # Needs refactoring
                payload_status[worker_key] = {
                    'response': response.status_code,
                    'message': sent_message['message']
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

                status = store_metrics_and_resources(
                    file_lock = file_lock,
                    type = 'resources',
                    subject = 'central',
                    area = 'network',
                    metrics = resource_metrics
                )
            except Exception as e:
                logger.error('Context sending error:' + str(e))
        
        successes = 0
        for worker_key in payload_status.keys():
            worker_data = payload_status[worker_key]
            if worker_data['response'] == 200 :
                # Check
                if worker_data['message'] == 'stored' or worker_data['message'] == 'ongoing jobs':
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
        central_resources_path = 'resources/experiment_' + str(current_experiment_number) + '/central.txt'
        central_resources = get_file_data(
            file_lock = file_lock,
            file_path = central_resources_path
        )
        # Potential info loss
        experiment_start = central_resources['general']['times']['experiment-time-start']
        experiment_end = time.time()
        experiment_total = experiment_end - experiment_start
        central_resources['general']['times']['experiment-time-end'] = experiment_end
        central_resources['general']['times']['experiment-total-seconds'] = experiment_total
        store_file_data(
            file_lock = file_lock,
            replace = True,
            file_folder_path = '',
            file_path = central_resources_path,
            data = central_resources
        )
            
    store_file_data(
        file_lock = file_lock,
        replace = True,
        file_folder_path = '',
        file_path = central_status_path,
        data = central_status
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

    status = store_metrics_and_resources(
        file_lock = file_lock,
        type = 'resources',
        subject = 'central',
        area = 'function',
        metrics = resource_metrics
    )
    
    return True