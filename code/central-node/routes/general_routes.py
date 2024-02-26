from flask import Blueprint, current_app, request, jsonify, render_template
import json

from functions.data_functions import *
from functions.model_functions import *
from functions.fed_functions import *

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    return 'Ok', 200
# Created and works
@general.route('/logs', methods=["GET"]) 
def central_logs():
    with open('logs/central.log', 'r') as f:
        logs = f.readlines()
    return render_template('logs.html', logs = logs)
# Created and works
@general.route('/training', methods=["GET"]) 
def training_status():
    training_status_path = 'logs/training_status.txt'
    training_status = None
    with open(training_status_path, 'r') as f:
        training_status = json.load(f)
    return jsonify({'status':training_status})
# Refactored and works
@general.route('/status', methods=["POST"])
def worker_status():
    received_worker_ip = request.remote_addr
    sent_worker_status = json.loads(request.json)

    set_worker_id, set_worker_ip, set_message = store_worker_status(
        worker_address = received_worker_ip,
        worker_status = sent_worker_status
    )

    return jsonify({'id': set_worker_id, 'address': set_worker_ip, 'message': set_message})  
# Refactored and works
@general.route('/models', methods=["GET"])
def stored_models():
    models = get_models()
    return jsonify({'models': models})
# Refactored and works
@general.route('/start', methods=["POST"])
def start_model_training():
    status = start_training()
    return 'Ok', 200
# Refactored and works
@general.route('/update', methods=["POST"]) 
def worker_update(): 
    sent_payload = json.loads(request.json)
    
    sent_worker_id = sent_payload['worker-id']
    sent_local_model = sent_payload['local-model']
    sent_cycle = sent_payload['cycle']
    sent_train_size = sent_payload['train-size']

    store_update(
        worker_id = sent_worker_id,
        local_model = sent_local_model,
        cycle = sent_cycle,
        train_size = sent_train_size 
    ) 
    
    return 'Ok', 200
# Refactored and works
@general.route('/predict', methods=["POST"])
def inference():
    sent_payload = json.loads(request.json)
    sent_input = sent_payload['input']
    sent_cycle = sent_payload['cycle']

    given_output = model_inference(
        input = sent_input,
        cycle = sent_cycle
    )

    return jsonify({'predictions': given_output})

