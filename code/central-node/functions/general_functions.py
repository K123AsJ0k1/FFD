from flask import current_app
import os
import json
from typing import Dict
import requests
import base64

from functions.data_functions import *

def get_debug_mode() -> list:
   return current_app.config['DEBUG']
# Works
def store_worker_ip(
   worker_ip: str
):
   log_path = 'logs/worker_ips.txt'
   
   worker_ips = []
   if os.path.exists(log_path):
      with open(log_path, 'r') as f:
         worker_ips = json.load(f)
   
   highest_worker_id = 0
   new_ip = True
   for dict in worker_ips:
      if worker_ip == dict['address']:
         new_ip = False
      if highest_worker_id < dict['id']:
         highest_worker_id = dict['id']
   
   if new_ip:
      worker_ips.append({
         'id': highest_worker_id + 1,
         'address': worker_ip,
         'status': 'waiting'
      })
      with open(log_path, 'w') as f:
         json.dump(worker_ips, f)

def send_context_to_workers():
   print('Send context')
   log_path = 'logs/worker_ips.txt'
   model_path = 'models/initial_model_parameters.pth'
   if not os.path.exists(log_path) or not os.path.exists(model_path):
      return False
   
   GLOBAL_PARAMETERS = current_app.config['GLOBAL_PARAMETERS']
   WORKER_PARAMETERS = current_app.config['WORKER_PARAMETERS']
   print('Get model')
   global_model = torch.load(model_path)

   worker_ips = None
   with open(log_path, 'r') as f:
      worker_ips = json.load(f)
   print('Get worker data')
   data_list, columns = split_data_between_workers(
      worker_amount = len(worker_ips)
   )
   
   formatted_global_model = {
      'weights': global_model['linear.weight'].numpy().tolist(),
      'bias': global_model['linear.bias'].numpy().tolist()
   }
   
   index = 0
   for dict in worker_ips:
      worker_address = 'http://' + dict['address'] + ':7500/context'
      #print(worker_address)
      worker_parameters = WORKER_PARAMETERS.copy()
      worker_parameters['address'] = dict['address']
      worker_parameters['id'] = dict['id']
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
            data = json_payload,
            headers = {
               'Content-type':'application/json', 
               'Accept':'application/json'
            }
         )
         print(response.status_code)
      except Exception as e:
        print(e)

      index = index + 1
   

      

   
