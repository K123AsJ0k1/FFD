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
