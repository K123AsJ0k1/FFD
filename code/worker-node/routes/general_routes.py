from flask import Blueprint, current_app, request, jsonify
import json

from functions.data_functions import *
from functions.model_functions import *

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    return 'Ok', 200
# Need st obe refactored
@general.route('/context', methods=["POST"]) 
def set_training_context():
    sent_payload = json.loads(request.json)
    
    sent_global_parameters = sent_payload['global-parameters']
    sent_worker_parameters = sent_payload['worker-parameters']
    sent_model = sent_payload['global-model']
    sent_worker_data = sent_payload['worker-data']
    
    status = store_training_context(
        global_parameters = sent_global_parameters,
        worker_parameters = sent_worker_parameters,
        global_model = sent_model,
        worker_data = sent_worker_data
    )

    return 'Ok', 200