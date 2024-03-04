from flask import Blueprint, jsonify, render_template
import json

from functions.general_functions import *

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    return 'Ok', 200
# Refactored and works
@general.route('/logs', methods=["GET"]) 
def get_central_logs():
    with open('logs/central.log', 'r') as f:
        logs = f.readlines()
    return render_template('logs.html', logs = logs)
# Refactor
@general.route('/training', methods=["GET"]) 
def get_training_status():
    training_status_path = 'logs/training_status.txt'
    training_status = None
    with open(training_status_path, 'r') as f:
        training_status = json.load(f)
    return jsonify({'status':training_status})
# Refactored and works
@general.route('/models', methods=["GET"])
def get_stored_models():
    models = get_models()
    return jsonify({'models': models})