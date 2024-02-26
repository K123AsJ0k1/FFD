from flask import Blueprint, current_app, request, jsonify, render_template
import json

from functions.data_functions import *
from functions.model_functions import *

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    return 'Ok', 200
# Created and works
@general.route('/logs', methods=["GET"]) 
def worker_logs():
    with open('logs/worker.log', 'r') as f:
        logs = f.readlines()
    return render_template('logs.html', logs = logs)
# Created
@general.route('/worker', methods=["GET"]) 
def worker_status():
    worker_status_path = 'logs/worker_status.txt'
    worker_status = None
    with open(worker_status_path, 'r') as f:
        worker_status = json.load(f)
    return jsonify({'status':worker_status})
# Refactored
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
    current_app.logger.warning('Local context:' + status)

    return 'Ok', 200