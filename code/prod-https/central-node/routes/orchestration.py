from flask import Blueprint, request, jsonify,current_app

import json

from functions.management.storage import store_worker
from functions.management.storage import store_update

orchestration = Blueprint('orchestration', __name__)
# Refactored and works
@orchestration.route('/status', methods=["POST"])
def worker_status():
    received_worker_ip = request.remote_addr
    sent_payload = json.loads(request.json)
    sent_status = sent_payload['status']

    payload = store_worker(
        file_lock = current_app.file_lock,
        logger = current_app.logger,
        minio_client = current_app.minio_client,
        prometheus_registry = current_app.prometheus_registry,
        prometheus_metrics = current_app.prometheus_metrics,
        address = received_worker_ip,
        status = sent_status
    )

    return jsonify(payload) 
# Refactored and works
@orchestration.route('/update', methods=["POST"]) 
def set_worker_update(): 
    sent_payload = json.loads(request.json)
    
    sent_worker_id = sent_payload['worker-id']
    sent_local_model = sent_payload['local-model']
    sent_experiment_name = sent_payload['experiment-name']
    sent_experiment = sent_payload['experiment']
    sent_cycle = sent_payload['cycle']

    payload = store_update(
        file_lock = current_app.file_lock,
        logger = current_app.logger,
        minio_client = current_app.minio_client,
        prometheus_registry = current_app.prometheus_registry,
        prometheus_metrics = current_app.prometheus_metrics,
        worker_id = sent_worker_id,
        model = sent_local_model,
        experiment_name = sent_experiment_name,
        experiment = sent_experiment,
        cycle = sent_cycle 
    ) 
        
    return jsonify(payload)