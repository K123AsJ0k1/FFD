from flask import Blueprint, current_app, request, jsonify

from functions.general_functions import *
from functions.data_functions import *
from functions.model_functions import *

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    local_model_training(cycle = 1)
    return 'Ok', 200

@general.route('/status', methods=["GET"]) 
def worker_status():
    return 'Ok', 200

@general.route('/context', methods=["POST"]) 
def set_training_context():
    sent_payload = request.json
    
    sent_global_parameters = sent_payload['global-parameters']
    sent_worker_parameters = sent_payload['worker-parameters']
    sent_model = sent_payload['global-model']
    sent_worker_data = sent_payload['worker-data']
    sent_data_dolumns = sent_payload['data-columns']
    sent_cycle = sent_payload['cycle']

    store_context(
        global_parameters = sent_global_parameters,
        worker_parameters = sent_worker_parameters,
        global_model = sent_model,
        worker_data = sent_worker_data,
        cycle = sent_cycle
    )

    preprocess_into_train_and_test_tensors(
        data_columns = sent_data_dolumns,
        cycle = sent_cycle
    )

    return 'Ok', 200