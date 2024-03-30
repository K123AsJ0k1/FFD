from flask import Blueprint, request, jsonify, current_app
import json
from functions.management.storage import store_update
from functions.management.pipeline import start_pipeline

pipeline = Blueprint('pipeline', __name__)
# Refactored and works
@pipeline.route('/start', methods=["POST"])
def start_training():
    sent_payload = json.loads(request.json)

    sent_experiment = sent_payload['experiment']
    sent_parameters = sent_payload['parameters']
    sent_data = sent_payload['data']
    sent_columns = sent_payload['columns']

    status = start_pipeline(
        file_lock = current_app.file_lock,
        logger = current_app.logger,
        mlflow_client = current_app.mlflow_client,
        minio_client = current_app.minio_client,
        experiment = sent_experiment,
        parameters = sent_parameters,
        df_data = sent_data,
        df_columns = sent_columns
    )
    #current_app.logger.info('Starting training: ' + str(status))
    return jsonify({'training': status})
# Refactored and works
@pipeline.route('/update', methods=["POST"]) 
def set_worker_update(): 
    sent_payload = json.loads(request.json)
    
    sent_worker_id = sent_payload['worker-id']
    sent_local_model = sent_payload['local-model']
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
        experiment = sent_experiment,
        cycle = sent_cycle 
    ) 
        
    return jsonify(payload)