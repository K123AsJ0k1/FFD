from flask import Blueprint, request, jsonify, current_app

import json

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
        minio_client = current_app.minio_client,
        mlflow_client = current_app.mlflow_client,
        prometheus_registry = current_app.prometheus_registry,
        prometheus_metrics = current_app.prometheus_metrics,
        experiment = sent_experiment,
        parameters = sent_parameters,
        df_data = sent_data,
        df_columns = sent_columns
    )
    current_app.logger.info('Start training: ' + str(status))

    return jsonify({'training': status})