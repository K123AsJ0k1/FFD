from flask import current_app
import os
import json
import requests

from functions.data_functions import *

# Works
def store_worker_ip(
   worker_ip: str
):
   worker_status_path = 'logs/worker_status.txt'
   
   worker_status = []
   if os.path.exists(worker_status_path):
      with open(worker_status_path, 'r') as f:
         worker_status = json.load(f)
   
   highest_worker_id = 0
   new_ip = True
   for dict in worker_status:
      if worker_ip == dict['address']:
         new_ip = False
      if highest_worker_id < dict['id']:
         highest_worker_id = dict['id']
   
   if new_ip:
      worker_status.append({
         'id': highest_worker_id + 1,
         'address': worker_ip,
         'status': 'waiting'
      })
      with open(worker_status_path, 'w') as f:
         json.dump(worker_status, f) 

def send_context_to_workers():
   GLOBAL_PARAMETERS = current_app.config['GLOBAL_PARAMETERS']
   WORKER_PARAMETERS = current_app.config['WORKER_PARAMETERS']
   
   worker_status_path = 'logs/worker_status.txt'
   if not os.path.exists(worker_status_path):
      return False
   
   worker_status = None
   with open(worker_status_path, 'r') as f:
      worker_status = json.load(f)
   
   files = os.listdir('models')
   current_cycle = 0
   for file in files:
      if 'worker' in file:
         first_split = file.split('.')
         second_split = first_split[0].split('_')
         cycle = int(second_split[2])
         
         if current_cycle < cycle:
               current_cycle = cycle

   global_model_path = 'models/global_model_' + str(current_cycle) + '.pth'
   
   global_model = torch.load(global_model_path)

   data_list, columns = split_data_between_workers(
      worker_amount = len(worker_status)
   )
   
   formatted_global_model = {
      'weights': global_model['linear.weight'].numpy().tolist(),
      'bias': global_model['linear.bias'].numpy().tolist()
   }
   
   index = 0
   for dict in worker_status:
      worker_address = 'http://' + dict['address'] + ':7500/context'
      
      worker_parameters = WORKER_PARAMETERS.copy()
      worker_parameters['address'] = dict['address']
      worker_parameters['worker-id'] = dict['id']
      worker_parameters['status'] = dict['status']
      worker_parameters['cycle'] = 1
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
   

      

   
