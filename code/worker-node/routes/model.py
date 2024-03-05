from flask import Blueprint, current_app, request, jsonify, render_template
import json

from functions.model import model_inference

model = Blueprint('model', __name__)

# Refactored and works
@model.route('/predict', methods=["POST"])
def inference():
    sent_payload = json.loads(request.json)
    sent_input = sent_payload['input']
    sent_cycle = sent_payload['cycle']

    given_output = model_inference(
        input = sent_input,
        cycle = sent_cycle
    )

    return jsonify({'predictions': given_output})