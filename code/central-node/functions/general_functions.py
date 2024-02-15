from flask import current_app
import os
import json
import requests

from functions.data_functions import *

'''
training status format:
- entry: dict
   - parameters: dict
      - cycle: int
      - global metrics: list
         - metrics: dict
            - loss: int,
            - confusion list
            - recall: int
            - selectivity: int
            - precision: int
            - miss-rate: int
            - fall-out: int
            - balanced-accuracy: int 
            - accuracy: int
   - workers: list
      - worker: dict
         - id: int 
         - address: str
         - status: str
         - local metrics: list
            - metrics: dict
               - loss: int,
               - confusion list
               - recall: int
               - selectivity: int
               - precision: int
               - miss-rate: int
               - fall-out: int
               - balanced-accuracy: int 
               - accuracy: int
'''
# Created
def initilize_training_status():
   training_status_path = 'logs/training_status.txt'
   if os.path.exists(training_status_path):
      return False
   
   training_status = {
      'parameters': {
         'cycle': 0,
         'global-metrics': []
      },
      'workers': []
   }
   
   with open(training_status_path, 'w') as f:
      json.dump(training_status, f) 
   return True
# Created
def store_global_metrics(
   metrics: any
) -> bool:
   training_status_path = 'logs/training_status.txt'
   if not os.path.exists(training_status_path):
      return False
   training_status = None
   with open(training_status_path, 'r') as f:
      training_status = json.load(f)
   training_status['parameters']['global-metrics'].append(metrics)
   with open(training_status_path, 'w') as f:
      json.dump(training_status, f) 
   return True
# refactored
def store_worker_status(
   worker_ip: str
) -> bool:
   training_status_path = 'logs/training_status.txt'
   
   training_status = None
   if not os.path.exists(training_status_path):
      return False
   
   with open(training_status_path, 'r') as f:
      training_status = json.load(f)
   
   highest_worker_id = 0
   new_ip = True
   for dict in training_status['workers']:
      if worker_ip == dict['address']:
         new_ip = False
      if highest_worker_id < dict['id']:
         highest_worker_id = dict['id']
   
   if new_ip:
      training_status['workers'].append({
         'id': highest_worker_id + 1,
         'address': worker_ip,
         'status': 'waiting',
         'metrics': {}
      })
      with open(training_status_path, 'w') as f:
         json.dump(training_status_path, f) 
   return True
# Refactored
def send_context_to_workers():
   GLOBAL_PARAMETERS = current_app.config['GLOBAL_PARAMETERS']
   WORKER_PARAMETERS = current_app.config['WORKER_PARAMETERS']
   
   training_status_path = 'logs/training_status.txt'
   if not os.path.exists(training_status_path):
      return False
   
   training_status = None
   with open(training_status_path, 'r') as f:
      training_status = json.load(f)
   
   global_model_path = 'models/global_model_' + str(training_status['parameters']['cycle']) + '.pth'
   if not os.path.exists(global_model_path):
      return False

   global_model = torch.load(global_model_path)
   data_list, columns = split_data_between_workers(
      worker_amount = len(training_status['workers'])
   )
   
   formatted_global_model = {
      'weights': global_model['linear.weight'].numpy().tolist(),
      'bias': global_model['linear.bias'].numpy().tolist()
   }
   
   index = 0
   for dict in training_status['workers']:
      worker_address = 'http://' + dict['address'] + ':7500/context'
      
      worker_parameters = WORKER_PARAMETERS.copy()
      worker_parameters['address'] = dict['address']
      worker_parameters['worker-id'] = dict['id']
      worker_parameters['status'] = dict['status']
      worker_parameters['cycle'] = training_status['parameters']['cycle'] + 1
      worker_parameters['columns'] = columns
      
      payload = {
         'worker-id': dict['id'],
         'global-parameters': GLOBAL_PARAMETERS,
         'worker-parameters': worker_parameters,
         'global-model': formatted_global_model,
         'worker-data': data_list[index]
      }

      json_payload = json.dumps(payload) 

      try:
         response = requests.post(
            url = worker_address, 
            json = json_payload,
            headers = {
               'Content-type':'application/json', 
               'Accept':'application/json'
            }
         )
      except Exception as e:
         current_app.logger.error('Context sending error')
         current_app.logger.error(e)

      index = index + 1
   

      

   
