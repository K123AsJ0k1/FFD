from flask import Blueprint, current_app, request, jsonify

from functions.general_functions import *
from functions.data_functions import *
from functions.model_functions import *

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    #send_context_to_workers()
    #split_data_between_workers(
    #    worker_amount = 4
    #)
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
    sent_payload = request.json
    print(sent_payload)
    print(sent_payload['local-model'])
    #print(sent_payload['worker-id'])
    #print(sent_payload)
    #sent_worker_id = sent_payload['worker-id']
    #sent_local_model = sent_payload['local-model']
    #sent_cycle = sent_payload['cycle']
    #sent_train_size = sent_payload['train-size']

    #print(sent_worker_id)
    #print(sent_cycle)
    #print(sent_train_size)
    
    return 'Ok', 200
