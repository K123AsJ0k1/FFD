from flask import Blueprint, current_app, request, jsonify
import json

from functions.data_functions import *
from functions.model_functions import *

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    return 'Ok', 200
# Refactored and works
@general.route('/status', methods=["POST"])
def worker_status():
    received_worker_ip = request.remote_addr
    sent_payload = json.loads(request.json)

    sent_worker_status = sent_payload['status']
    sent_worker_id = sent_payload['id']

    set_worker_id, set_message = store_worker_status(
        worker_id = sent_worker_id,
        worker_ip = received_worker_ip,
        worker_status = sent_worker_status
    )

    return jsonify({'id': set_worker_id, 'message': set_message})  
# Refactored and works
@general.route('/start', methods=["POST"])
def start_model_training():
    status = central_worker_data_split()
    current_app.logger.warning('Global data split:' + str(status))
    
    status = preprocess_into_train_test_and_evaluate_tensors()
    current_app.logger.warning('Global preprocessing:' + str(status))
    
    status = initial_model_training()
    current_app.logger.warning('Global training:' + str(status))
    
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
# Need refactoring
@general.route('/predict', methods=["POST"])
def inference():
    sent_payload = json.loads(request.json)
    sent_input = sent_payload['input']

    given_output = model_inference(
        input = sent_input
    )

    return jsonify({'predictions': given_output})
