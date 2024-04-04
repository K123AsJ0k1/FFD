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
        mlflow_client = current_app.mlflow_client,
        minio_client = current_app.minio_client,
        experiment = sent_experiment,
        parameters = sent_parameters,
        df_data = sent_data,
        df_columns = sent_columns
    )
    return jsonify({'training': status})