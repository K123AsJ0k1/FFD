from flask import current_app

import numpy as np
import pandas as pd
import os 
import json
import requests

from functions.general import get_current_experiment_number,get_current_global_model
from functions.storage import store_worker
# Refactor
def send_context_to_workers(
    logger: any
) -> bool:
    current_experiment_number = get_current_experiment_number()
    status_folder_path = 'status/experiment_' + str(current_experiment_number)
    central_status_path = status_folder_path + '/central.txt'
    if not os.path.exists(central_status_path):
        return False
    
    central_status = None
    with open(central_status_path, 'r') as f:
        central_status = json.load(f)

    if not central_status['start']:
        return False

    if not central_status['worker-split']:
        return False
    
    if not central_status['trained']:
        return False
    
    if central_status['sent']:
        return False
    
    os.environ['STATUS'] = 'sending'

    workers_status_path = status_folder_path + '/workers.txt'
    if not os.path.exists(workers_status_path):
        return False
    
    workers_status = None
    with open(workers_status_path, 'r') as f:
        workers_status = json.load(f)
    
    parameters_folder_path = 'parameters/experiment_' + str(current_experiment_number)
    central_parameters_path = parameters_folder_path + '/central.txt'
    if not os.path.exists(central_parameters_path):
        return False
    
    central_parameters = None
    with open(central_parameters_path, 'r') as f:
        central_parameters = json.load(f)

    model_parameters_path = parameters_folder_path + '/model.txt'
    if not os.path.exists(model_parameters_path):
        return False
    
    model_parameters = None
    with open(model_parameters_path, 'r') as f:
        model_parameters = json.load(f)

    worker_parameters_path = parameters_folder_path + '/worker.txt'
    if not os.path.exists(worker_parameters_path):
        return False
    
    worker_parameters = None
    with open(worker_parameters_path, 'r') as f:
        worker_parameters = json.load(f)

    global_model = get_current_global_model()
    formatted_global_model = {
        'weights': global_model['linear.weight'].numpy().tolist(),
        'bias': global_model['linear.bias'].numpy().tolist()
    }

    payload_status = {}

    data_folder_path = 'data/experiment_' + str(current_experiment_number)
    data_files = os.listdir(data_folder_path)
    for worker_key in workers_status.keys():
        worker_metadata = workers_status[worker_key]
        
        if not worker_metadata['status'] == 'waiting':
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
            #print(data_path)
            #data_path = data_folder_path + '/worker_' + worker_key + '_' + str(central_status['cycle']) + '.csv'
            worker_df = pd.read_csv(data_path)
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
            
            payload_status[worker_key] = {
                'response': response.status_code,
                'worker-address': worker_metadata['worker-address'],
                'status': worker_metadata['status']
            }
        except Exception as e:
            logger.error('Context sending error:' + str(e))

    successes = 0
    for worker_key in payload_status.keys():
        worker_data = payload_status[worker_key]
        if not worker_data['response'] == 200: 
            store_worker(
                address = worker_data['address'],
                status = worker_data['status']
            )
            continue
        successes = successes + 1 
    
    if not central_status['complete']:
        if not central_parameters['min-update-amount'] <= successes:
            return False
        os.environ['STATUS'] = 'waiting updates'
    else:
        os.environ['STATUS'] = 'training complete'
    
    central_status['sent'] = True
    with open(central_status_path, 'w') as f:
        json.dump(central_status, f, indent=4)

    return True