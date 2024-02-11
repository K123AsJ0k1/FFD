from flask import current_app
import os
import json
from typing import Dict

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
   for ip in worker_ips:
      if worker_ip == ip:
         new_ip = False
   
   if new_ip:
      worker_ips.append(worker_ip)
      with open(log_path, 'w') as f:
         json.dump(worker_ips, f)

def send_context_to_workers():
   log_path = 'logs/worker_ips.txt'
   if not os.path.exists(log_path):
      return False
   
   GLOBAL_SEED = current_app.config['GLOBAL_SEED']
   GLOBAL_USED_COLUMNS = current_app.config['GLOBAL_USED_COLUMNS']
   GLOBAL_TARGET_COLUMN = current_app.config['GLOBAL_TARGET_COLUMN']
   GLOBAL_SCALED_COLUMNS = current_app.config['GLOBAL_SCALED_COLUMNS']

   GLOBAL_INPUT_SIZE = current_app.config['GLOBAL_INPUT_SIZE'] 
   
   GLOBAL_LEARNING_RATE = current_app.config['GLOBAL_LEARNING_RATE'] 
   GLOBAL_SAMPLE_RATE = current_app.config['GLOBAL_SAMPLE_RATE']
   GLOBAL_MODEL_OPTIMIZER = current_app.config['GLOBAL_MODEL_OPTIMIZER']
   GLOBAL_TRAINING_EPOCHS = current_app.config['GLOBAL_TRAINING_EPOCHS']

   
   
   
      
   

   worker_ips = None
   with open(log_path, 'r') as f:
      worker_ips = json.load(f)

   
