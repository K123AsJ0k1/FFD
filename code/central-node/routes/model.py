from flask import Blueprint, request, jsonify, current_app
import json

from functions.model import model_inference

model = Blueprint('model', __name__)
# Refactored and works
@model.route('/predict', methods=["POST"])
def inference():
    sent_payload = json.loads(request.json)
    sent_experiment = sent_payload['experiment-id']
    sent_subject = sent_payload['subject']
    sent_cycle = sent_payload['cycle']
    sent_input = sent_payload['input']

    given_output = model_inference(
        file_lock = current_app.file_lock,
        experiment = sent_experiment,
        subject = sent_subject,
        cycle = sent_cycle,
        input = sent_input
    )

    return jsonify({'predictions': given_output})