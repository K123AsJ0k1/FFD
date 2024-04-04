from flask import Blueprint, request, jsonify, current_app
import json
from functions.training.model import model_inference

model = Blueprint('model', __name__)
# Refactor
@model.route('/predict', methods=["POST"]) 
def inference():
    sent_payload = json.loads(request.json)
    sent_experiment_name = sent_payload['experiment-name']
    sent_experiment = sent_payload['experiment']
    sent_cycle = sent_payload['cycle']
    sent_input = sent_payload['input']

    given_output = model_inference(
        file_lock = current_app.file_lock,
        logger = current_app.logger,
        minio_client = current_app.minio_client,
        prometheus_registry = current_app.prometheus_registry,
        prometheus_metrics = current_app.prometheus_metrics,
        experiment_name = sent_experiment_name,
        experiment = sent_experiment,
        cycle = sent_cycle,
        input = sent_input
    )

    return jsonify({'predictions': given_output})