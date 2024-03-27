from flask import Blueprint, current_app, request, jsonify, render_template
import json
from prometheus_client import generate_latest

from functions.general import get_worker_logs

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    return 'Ok', 200

@general.route('/metrics') 
def metrics():
    return generate_latest(current_app.prometheus_registry)
# Created and works
@general.route('/logs', methods=["GET"]) 
def worker_logs():
    current_logs = get_worker_logs()
    return render_template('logs.html', logs = current_logs)
