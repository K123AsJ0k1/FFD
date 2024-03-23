from flask import Blueprint, jsonify, render_template, request, current_app
import json
from functions.general import get_models, get_central_logs, get_metrics_resources_and_status, get_directory_and_file_sizes
from prometheus_client import generate_latest

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    return 'Ok', 200
@general.route('/metrics') 
def metrics():
    return generate_latest(current_app.prometheus_registry)
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
        file_lock = current_app.file_lock,
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
        file_lock = current_app.file_lock,
        experiment = sent_experiment,
        subject = sent_subject
    )
    return jsonify(data)
# Refactored and works
@general.route('/files', methods=["GET"])
def stored_file():
    data = get_directory_and_file_sizes()
    return jsonify(data)