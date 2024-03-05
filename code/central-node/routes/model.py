from flask import Blueprint, request, jsonify
import json

from functions.model import model_inference

model = Blueprint('model', __name__)
# Refactor
@model.route('/predict', methods=["POST"])
def model_inference():
    sent_payload = json.loads(request.json)
    sent_input = sent_payload['input']
    sent_cycle = sent_payload['cycle']

    given_output = model_inference(
        input = sent_input,
        cycle = sent_cycle
    )

    return jsonify({'predictions': given_output})