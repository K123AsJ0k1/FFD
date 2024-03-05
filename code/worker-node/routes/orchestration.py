from flask import Blueprint, current_app, request, jsonify, render_template
import json

from functions.data import *
from functions.model import *

orchestration = Blueprint('orchestration', __name__)

@orchestration.route('/point', methods=["POST"]) 
def set_point_to_central():
    sent_payload = json.loads(request.json)
    print(sent_payload)
     
    sent_central_address = sent_payload['central-address']

    worker_status_path = 'status/worker.txt'
    worker_status = None
    with open(worker_status_path, 'r') as f:
        worker_status = json.load(f)
    
    worker_status['central-address'] = sent_central_address
    with open(worker_status_path, 'w') as f:
        json.dump(worker_status, f, indent=4)

    return 'Ok', 200