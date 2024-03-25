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