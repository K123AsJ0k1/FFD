from flask import Blueprint, current_app, request, jsonify

import json

from functions.management.storage import store_training_context

orchestration = Blueprint('orchestration', __name__)

# Refactored and works
@orchestration.route('/context', methods=["POST"]) 
def set_training_context():
    sent_payload = json.loads(request.json)
     
    sent_info = sent_payload['info']
    sent_global_model = sent_payload['global-model']
    sent_worker_data_list = sent_payload['worker-data-list']
    sent_worker_data_columns = sent_payload['worker-data-columns']
 
    payload = store_training_context(
        file_lock = current_app.file_lock,
        logger = current_app.logger,
        minio_client = current_app.minio_client,
        mlflow_client = current_app.mlflow_client,
        prometheus_registry = current_app.prometheus_registry,
        prometheus_metrics = current_app.prometheus_metrics,
        info = sent_info,
        global_model = sent_global_model,
        df_data = sent_worker_data_list,
        df_columns = sent_worker_data_columns
    )
    
    return jsonify(payload)