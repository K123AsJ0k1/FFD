from flask import Blueprint, current_app, request, jsonify
import json

from functions.general_functions import *
from functions.data_functions import *
from functions.model_functions import *

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    update_global_model()
    return 'Ok', 200
# Works
@general.route('/register', methods=["POST"])
def worker_registration():
    received_worker_ip = request.remote_addr
    store_worker_ip(
        worker_ip = received_worker_ip
    )
    return 'Ok', 200  

@general.route('/start', methods=["POST"])
def start_model_training():
    split_status = central_worker_data_split()
    tensor_status = preprocess_into_train_test_and_evaluate_tensors()
    model_status = initial_model_training()
    print(split_status,tensor_status,model_status)
    send_context_to_workers()
    return 'Ok', 200

@general.route('/inference', methods=["POST"]) 
def model_inference():
    return 'Ok', 200

@general.route('/update', methods=["POST"]) 
def worker_update():
    # Wierd type error, which was fixed with loading 
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
