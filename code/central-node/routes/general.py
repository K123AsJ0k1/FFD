from flask import Blueprint, jsonify, render_template, request
import json

from functions.general import get_models, get_central_logs, get_metrics_resources_and_status

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    return 'Ok', 200
# Refactored and works
@general.route('/logs', methods=["GET"]) 
def central_logs():
    current_logs = get_central_logs()
    return render_template('logs.html', logs = current_logs)
# Created and works
@general.route('/storage', methods=["GET"])
def intrastructure_metrics_resources_and_status():
    sent_payload = json.loads(request.json)

    sent_type = sent_payload['type']
    sent_experiment = sent_payload['experiment']
    sent_subject = sent_payload['subject']

    data = get_metrics_resources_and_status(
        type = sent_type,
        experiment = sent_experiment,
        subject = sent_subject
    )
    return jsonify(data)
# Refactored and works
@general.route('/models', methods=["GET"])
def stored_models():
    sent_payload = json.loads(request.json)

    sent_experiment = sent_payload['experiment']
    sent_subject = sent_payload['subject']

    data = get_models(
        experiment = sent_experiment,
        subject = sent_subject
    )
    return jsonify(data)