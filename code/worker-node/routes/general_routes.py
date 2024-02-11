from flask import Blueprint, current_app, request, jsonify

from functions.general_functions import *

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    return 'Ok', 200

@general.route('/status', methods=["GET"]) 
def worker_status():
    return 'Ok', 200

@general.route('/context', methods=["POST"]) 
def set_training_context():
    sent_payload = request.json
    #print(sent_payload)
    sent_global_parameters = sent_payload['global-parameters']
    sent_worker_parameters = sent_payload['worker-parameters']
    sent_model = sent_payload['global-model']
    #sent_worker_data = sent_payload['worker-data']

    print(sent_global_parameters)
    print(sent_worker_parameters)
    print(sent_model)

    return 'Ok', 200