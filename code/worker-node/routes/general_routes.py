from flask import Blueprint, current_app, request, jsonify

from functions.general_functions import *

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    return 'Ok', 200

@general.route('/status', methods=["GET"]) 
def worker_status():
    return 'Ok', 200

@general.route('/context', methods=["GET"]) 
def set_training_context():
    sent_payload = request.json
    sent_parameters = sent_payload['global_model']
    sent_hyperparameters = sent_payload['global_hyperparameters']
    sent_worker_data = sent_payload['worker_data']
    return 'Ok', 200