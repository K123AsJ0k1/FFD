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
   
   new_ip = True
   for dict in worker_ips:
      if worker_ip == dict['address']:
         new_ip = False
   
   if new_ip:
      worker_ips.append({
         'address': worker_ip,
         'status': 'ready'
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
   pickled_data_list = split_data_between_workers(
      worker_amount = len(worker_ips)
   )
   print(global_model)
   #print(global_model.numpy())

   formatted_global_model = {
      'weights': global_model['linear.weight'].numpy().tolist(),
      'bias': global_model['linear.bias'].numpy().tolist()
   }

   #print(formatted_global_model)

   index = 0
   for dict in worker_ips:
      worker_address = 'http://' + dict['address'] + ':7500/context'
      print(worker_address)
      payload = {
         'global-parameters': GLOBAL_PARAMETERS,
         'worker-parameters': WORKER_PARAMETERS,
         'global-model': formatted_global_model
      }

      #'worker-data': base64.b64encode(pickled_data_list[index]).decode('utf-8')
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
   

      

   
