from flask import Blueprint, current_app, request, jsonify, render_template
import json
from functions.general import get_metrics_resources_and_status, get_worker_logs, get_directory_and_file_sizes

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    return 'Ok', 200
# Created and works
@general.route('/logs', methods=["GET"]) 
def worker_logs():
    current_logs = get_worker_logs()
    return render_template('logs.html', logs = current_logs)
# Created and works
@general.route('/storage', methods=["GET"])
def worker_metrics_resources_and_status():
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
@general.route('/files', methods=["GET"])
def stored_file():
    data = get_directory_and_file_sizes()
    return jsonify(data)