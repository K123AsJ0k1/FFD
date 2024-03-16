import os
import json
import requests
import psutil
import time
from datetime import datetime

from functions.general import get_current_experiment_number, get_file_data, get_wanted_model
from functions.storage import store_metrics_and_resources, store_file_data

# Refactored and works
def send_info_to_central(
    file_lock: any,
    logger: any
) -> bool:
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    # In this simulated infrastructure we will assume that workers can failure restart in such a way that files are lost
    current_experiment_number = get_current_experiment_number()

    worker_status_path = 'status/templates/worker.txt'
    local_metrics_path = None
    worker_resources_path = None
    if not current_experiment_number == 0:
        worker_folder_path = 'status/experiment_' + str(current_experiment_number) 
        metrics_folder_path = 'metrics/experiment_' + str(current_experiment_number)
        resource_folder_path = 'resources/experiment_' + str(current_experiment_number)
        
        worker_status_path = worker_folder_path + '/worker.txt'
        local_metrics_path = metrics_folder_path + '/local.txt'
        worker_resources_path = resource_folder_path + '/worker.txt'

    worker_status = get_file_data(
        file_lock = file_lock,
        file_path = worker_status_path
    )
    
    if worker_status is None:
        return False

    if worker_status['central-address'] == '':
        return False
    
    local_metrics = get_file_data(
        file_lock = file_lock,
        file_path = local_metrics_path
    )

    worker_resources = get_file_data(
        file_lock = file_lock,
        file_path = worker_resources_path
    )

    worker_status['status'] = os.environ.get('STATUS')
    logger.info('Sending status to central')
    info = {
        'status': worker_status,
        'metrics': {
            'local': local_metrics,
            'resources': worker_resources
        }
    }
    # key order changes
    # Refactor to have failure fixing
    address = worker_status['central-address'] + '/status'
    
    payload = json.dumps(info) 
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
                
                worker_folder_path = 'status/experiment_' + str(current_experiment_number) 
                metrics_folder_path = 'metrics/experiment_' + str(current_experiment_number)
                resource_folder_path = 'resources/experiment_' + str(current_experiment_number)

                worker_status_path = worker_folder_path + '/worker.txt'
                local_metrics_path = metrics_folder_path + '/local.txt'
                worker_resources_path = resource_folder_path + '/worker.txt'
                
                if message == 'registered':
                    local_metrics = sent_payload['metrics']['local']
                    worker_resources = sent_payload['metrics']['resources']
                    if not len(local_metrics) == 0: 
                        store_file_data(
                            file_lock = file_lock,
                            replace = True,
                            file_folder_path = metrics_folder_path,
                            file_path = local_metrics_path,
                            data = local_metrics
                        )
                    else:
                        store_file_data(
                            file_lock = file_lock,
                            replace = True,
                            file_folder_path = metrics_folder_path,
                            file_path = local_metrics_path,
                            data = {}
                        )
                    if not len(worker_resources) == 0:
                        store_file_data(
                            file_lock = file_lock,
                            replace = True,
                            file_folder_path = resource_folder_path,
                            file_path = worker_resources_path,
                            data = worker_resources
                        )
                    else:
                        resource_template = {
                            'general': {
                                'physical-cpu-amount': psutil.cpu_count(logical=False),
                                'total-cpu-amount': psutil.cpu_count(logical=True),
                                'min-cpu-frequency-mhz': psutil.cpu_freq().min,
                                'max-cpu-frequency-mhz': psutil.cpu_freq().max,
                                'total-ram-amount-bytes': psutil.virtual_memory().total,
                                'available-ram-amount-bytes': psutil.virtual_memory().free,
                                'total-disk-amount-bytes': psutil.disk_usage('.').total,
                                'available-disk-amount-bytes': psutil.disk_usage('.').free,
                                'times': {
                                    'experiment-date': 0,
                                    'experiment-time-start': 0,
                                    'experiment-time-end': 0,
                                    'experiment-total-seconds':0
                                }
                            },
                            'function': {},
                            'network': {},
                            'training': {},
                            'inference': {}
                        }
                        store_file_data(
                            file_lock = file_lock,
                            replace = True,
                            file_folder_path = resource_folder_path,
                            file_path = worker_resources_path,
                            data = resource_template
                        )
                if message == 'experiment':
                    store_file_data(
                        file_lock = file_lock,
                        replace = True,
                        file_folder_path = metrics_folder_path,
                        file_path = local_metrics_path,
                        data = {}
                    )

                    resource_template = {
                        'general': {
                            'physical-cpu-amount': psutil.cpu_count(logical=False),
                            'total-cpu-amount': psutil.cpu_count(logical=True),
                            'min-cpu-frequency-mhz': psutil.cpu_freq().min,
                            'max-cpu-frequency-mhz': psutil.cpu_freq().max,
                            'total-ram-amount-bytes': psutil.virtual_memory().total,
                            'available-ram-amount-bytes': psutil.virtual_memory().free,
                            'total-disk-amount-bytes': psutil.disk_usage('.').total,
                            'available-disk-amount-bytes': psutil.disk_usage('.').free,
                            'times': {
                                'experiment-date': 0,
                                'experiment-time-start': 0,
                                'experiment-time-end': 0,
                                'experiment-total-seconds':0,
                                '1':{
                                    'cycle-time-start':0,
                                    'cycle-time-end':0,
                                    'cycle-total-seconds':0
                                }
                            }
                        },
                        'function': {},
                        'network': {},
                        'training': {},
                        'inference': {}
                    }
                    store_file_data(
                        file_lock = file_lock,
                        replace = True,
                        file_folder_path = resource_folder_path,
                        file_path = worker_resources_path,
                        data = resource_template
                    )

                    worker_status_template_path = 'status/templates/worker.txt'
                    worker_status = get_file_data(
                        file_lock = file_lock,
                        file_path = worker_status_template_path
                    )
                    
            # Worker address has changed
            if message == 'rerouted':
                worker_status['id'] = sent_payload['status']['id']

            if not message == 'checked':
                store_file_data(
                    file_lock = file_lock,
                    replace = True,
                    file_folder_path = worker_folder_path,
                    file_path = worker_status_path,
                    data = worker_status
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

            status = store_metrics_and_resources(
                file_lock = file_lock,
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

        status = store_metrics_and_resources(
            file_lock = file_lock,
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
    file_lock: any,
    logger: any
) -> bool:  
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    current_experiment_number = get_current_experiment_number()

    worker_status_path = 'status/experiment_' + str(current_experiment_number) + '/worker.txt'
    worker_status = get_file_data(
        file_lock = file_lock,
        file_path = worker_status_path
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

    local_model = get_wanted_model(
        file_lock = file_lock,
        experiment = current_experiment_number,
        subject = 'local',
        cycle = worker_status['cycle']
    )

    formatted_local_model = {
      'weights': local_model['linear.weight'].numpy().tolist(),
      'bias': local_model['linear.bias'].numpy().tolist()
    }

    update = {
        'worker-id': str(worker_status['id']),
        'local-model': formatted_local_model,
        'cycle': worker_status['cycle']
    }
    
    central_url = worker_status['central-address'] + '/update'
    payload = json.dumps(update)
    # If an error happens in central, update might not be received
    success = False
    for i in range(0,10):
        if success:
            break

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
                success = True

                worker_resources_path = 'resources/experiment_' + str(current_experiment_number) + '/worker.txt'
                worker_resources = get_file_data(
                    file_lock = file_lock,
                    file_path = worker_resources_path
                )
                # Potential info loss
                cycle_start = worker_resources['general']['times'][str(worker_status['cycle'])]['cycle-time-start']
                cycle_end = time.time()
                cycle_total = cycle_end-cycle_start
                worker_resources['general']['times'][str(worker_status['cycle'])]['cycle-time-end'] = cycle_end
                worker_resources['general']['times'][str(worker_status['cycle'])]['cycle-total-seconds'] = cycle_total
                store_file_data(
                    file_lock = file_lock,
                    replace = True,
                    file_folder_path = '',
                    file_path = worker_resources_path,
                    data = worker_resources
                )

                worker_status['updated'] = True
                worker_status['stored'] = False
                store_file_data(
                    file_lock = file_lock,
                    replace = True,
                    file_folder_path = '',
                    file_path = worker_status_path,
                    data = worker_status
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

                status = store_metrics_and_resources(
                    file_lock = file_lock,
                    type = 'resources',
                    subject = 'worker',
                    area = 'network',
                    metrics = resource_metrics
                )
                return True
            
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

            status = store_metrics_and_resources(
                file_lock = file_lock,
                type = 'resources',
                subject = 'worker',
                area = 'network',
                metrics = resource_metrics
            )
            return False
        except Exception as e:
            logger.error('Status sending error:' + str(e))
            return False