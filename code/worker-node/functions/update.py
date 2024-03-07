from flask import current_app

import torch  
import os
import json
import requests
import psutil
import time

from functions.general import get_current_experiment_number
from functions.storage import store_metrics_and_resources

# Refactored and works
def send_info_to_central(
    logger: any
) -> bool:
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    storage_folder_path = 'storage'
    # In this simulated infrastructure we will assume that workers can failure restart in such a way that files are lost
    current_experiment_number = get_current_experiment_number()

    worker_status_path = None
    local_metrics_path = None
    worker_resources_path = None
    if current_experiment_number == 0:
        worker_status_path = storage_folder_path + '/status/templates/worker.txt'
    else:
        worker_folder_path = storage_folder_path + '/status/experiment_' + str(current_experiment_number) 
        metrics_folder_path = storage_folder_path + '/metrics/experiment_' + str(current_experiment_number)
        resource_folder_path = storage_folder_path + '/resources/experiment_' + str(current_experiment_number)

        os.makedirs(worker_folder_path, exist_ok=True)
        os.makedirs(metrics_folder_path, exist_ok=True)
        os.makedirs(resource_folder_path, exist_ok=True)

        worker_status_path = worker_folder_path + '/worker.txt'
        local_metrics_path = metrics_folder_path + '/local.txt'
        worker_resources_path = resource_folder_path + '/worker.txt'

    if not os.path.exists(worker_status_path):
        return False

    worker_status = None
    with open(worker_status_path, 'r') as f:
        worker_status = json.load(f)

    if worker_status['central-address'] == '':
        return False

    local_metrics = None
    if os.path.exists(worker_resources_path):
        with open(local_metrics_path, 'r') as f:
            local_metrics = json.load(f)

    worker_resources = None
    if os.path.exists(worker_resources_path):
        with open(worker_resources_path, 'r') as f:
            worker_resources = json.load(f)
    
    worker_status['status'] = os.environ.get('STATUS')

    info = {
        'status': worker_status,
        'metrics': {
            'local': local_metrics,
            'resources': worker_resources
        }
    }
    # key order changes
    payload = json.dumps(info) 
    address = worker_status['central-address'] + '/status'
    try:
        response = requests.post(
            url = address,
            json = payload,
            headers = {
               'Content-type':'application/json', 
               'Accept':'application/json'
            }
        )

        if response.status_code == 200:
            sent_payload = json.loads(response.text)
            
            message = sent_payload['message']
            logger.info('Central message: ' + message)
            # Worker is either new or it has failure restated
            if message == 'registered' or message == 'experiment':
                worker_status = sent_payload['status']
                current_experiment_number = sent_payload['experiment_id']
                
                worker_folder_path = storage_folder_path + '/status/experiment_' + str(current_experiment_number) 
                metrics_folder_path = storage_folder_path + '/metrics/experiment_' + str(current_experiment_number)
                resource_folder_path = storage_folder_path + '/resources/experiment_' + str(current_experiment_number)

                os.makedirs(worker_folder_path, exist_ok=True)
                os.makedirs(metrics_folder_path, exist_ok=True)
                os.makedirs(resource_folder_path, exist_ok=True)

                worker_status_path = worker_folder_path + '/worker.txt'
                local_metrics_path = metrics_folder_path + '/local.txt'
                worker_resources_path = resource_folder_path + '/worker.txt'
                
                if message == 'registered':
                    local_metrics = sent_payload['metrics']['local']
                    worker_resources = sent_payload['metrics']['resources']
                    if not len(local_metrics) == 0: 
                        with open(local_metrics_path, 'w') as f:
                            json.dump(local_metrics, f, indent=4)
                    if not len(worker_resources) == 0:
                        with open(worker_resources_path, 'w') as f:
                            json.dump(worker_resources, f, indent=4)
                if message == 'experiment':
                    with open(local_metrics_path, 'w') as f:
                        json.dump({}, f, indent=4)

                    worker_status_template_path = storage_folder_path + '/status/templates/worker.txt'
                    with open(worker_status_template_path, 'r') as f:
                        worker_status = json.load(f)
                    
            # Worker address has changed
            if message == 'rerouted':
                worker_status['id'] = sent_payload['status']['id']

            with open(worker_status_path, 'w') as f:
                json.dump(worker_status, f, indent=4)
            
            time_end = time.time()
            cpu_end = this_process.cpu_percent(interval=0.2)
            mem_end = psutil.virtual_memory().used 
            disk_end = psutil.disk_usage('.').used

            time_diff = (time_end - time_start) 
            cpu_diff = cpu_end - cpu_start 
            mem_diff = (mem_end - mem_start) / (1024 ** 2) 
            disk_diff = (disk_end - disk_start) / (1024 ** 2) 

            resource_metrics = {
                'name': 'sending-info-to-central',
                'status-code': response.status_code,
                'processing-time-seconds': time_diff,
                'elapsed-time-seconds': response.elapsed.total_seconds(),
                'cpu-percentage': cpu_diff,
                'ram-megabytes': mem_diff,
                'disk-megabytes': disk_diff
            }

            status = store_metrics_and_resources(
                type = 'resources',
                subject = 'worker',
                area = 'network',
                metrics = resource_metrics
            )
            
            return True
        
        time_end = time.time()
        cpu_end = this_process.cpu_percent(interval=0.2)
        mem_end = psutil.virtual_memory().used 
        disk_end = psutil.disk_usage('.').used

        time_diff = (time_end - time_start) 
        cpu_diff = cpu_end - cpu_start 
        mem_diff = (mem_end - mem_start) / (1024 ** 2) 
        disk_diff = (disk_end - disk_start) / (1024 ** 2) 

        resource_metrics = {
            'name': 'sending-info-to-central',
            'status-code': response.status_code,
            'processing-time-seconds': time_diff,
            'elapsed-time-seconds': response.elapsed.total_seconds(),
            'cpu-percentage': cpu_diff,
            'ram-megabytes': mem_diff,
            'disk-megabytes': disk_diff
        }

        status = store_metrics_and_resources(
            type = 'resources',
            subject = 'worker',
            area = 'network',
            metrics = resource_metrics
        )
        return False
    except Exception as e:
        logger.error('Sending info to central error:' +  str(e)) 
        return False
# Refactored and works
def send_update_to_central(
    logger: any
) -> bool:  
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    storage_folder_path = 'storage'
    current_experiment_number = get_current_experiment_number()

    worker_status_path = storage_folder_path + '/status/experiment_' + str(current_experiment_number) + '/worker.txt'
    if not os.path.exists(worker_status_path):
        return False
    
    worker_status = None
    with open(worker_status_path, 'r') as f:
        worker_status = json.load(f)

    if not worker_status['stored'] or not worker_status['preprocessed'] or not worker_status['trained']:
        return False

    if worker_status['complete']:
        return False

    if worker_status['updated']:
        return False

    os.environ['STATUS'] = 'updating'

    local_model_path = storage_folder_path + '/models/experiment_' + str(current_experiment_number) + '/local_' + str(worker_status['cycle']) + '.pth'
    local_model = torch.load(local_model_path)

    formatted_local_model = {
      'weights': local_model['linear.weight'].numpy().tolist(),
      'bias': local_model['linear.bias'].numpy().tolist()
    }

    update = {
        'worker-id': str(worker_status['id']),
        'local-model': formatted_local_model,
        'cycle': worker_status['cycle']
    }
    
    payload = json.dumps(update)
    central_url = worker_status['central-address'] + '/update'
    try:
        response = requests.post(
            url = central_url, 
            json = payload,
            headers = {
                'Content-type':'application/json', 
                'Accept':'application/json'
            }
        )
        if response.status_code == 200:
            worker_status['updated'] = True
            with open(worker_status_path, 'w') as f:
                json.dump(worker_status, f, indent=4)
            os.environ['STATUS'] = 'waiting'

            time_end = time.time()
            cpu_end = this_process.cpu_percent(interval=0.2)
            mem_end = psutil.virtual_memory().used 
            disk_end = psutil.disk_usage('.').used

            time_diff = (time_end - time_start) 
            cpu_diff = cpu_end - cpu_start 
            mem_diff = (mem_end - mem_start) / (1024 ** 2) 
            disk_diff = (disk_end - disk_start) / (1024 ** 2) 

            resource_metrics = {
                'name': 'sending-update-to-central',
                'status-code': response.status_code,
                'processing-time-seconds': time_diff,
                'elapsed-time-seconds': response.elapsed.total_seconds(),
                'cpu-percentage': cpu_diff,
                'ram-megabytes': mem_diff,
                'disk-megabytes': disk_diff
            }

            status = store_metrics_and_resources(
                type = 'resources',
                subject = 'worker',
                area = 'network',
                metrics = resource_metrics
            )
            return True
        
        time_end = time.time()
        cpu_end = this_process.cpu_percent(interval=0.2)
        mem_end = psutil.virtual_memory().used 
        disk_end = psutil.disk_usage('.').used

        time_diff = (time_end - time_start) 
        cpu_diff = cpu_end - cpu_start 
        mem_diff = (mem_end - mem_start) / (1024 ** 2) 
        disk_diff = (disk_end - disk_start) / (1024 ** 2) 

        resource_metrics = {
            'name': 'sending-update-to-central',
            'status-code': response.status_code,
            'processing-time-seconds': time_diff,
            'elapsed-time-seconds': response.elapsed.total_seconds(),
            'cpu-percentage': cpu_diff,
            'ram-megabytes': mem_diff,
            'disk-megabytes': disk_diff
        }

        status = store_metrics_and_resources(
            type = 'resources',
            subject = 'worker',
            area = 'network',
            metrics = resource_metrics
        )
        return False
    except Exception as e:
        logger.error('Status sending error:' + str(e))
        return False