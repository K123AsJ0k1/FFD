from flask import Blueprint, current_app, request, jsonify, render_template
import json

from functions.data import *
from functions.model import *

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    return 'Ok', 200
# Created and works
@general.route('/logs', methods=["GET"]) 
def worker_logs():
    with open('logs/worker.log', 'r') as f:
        logs = f.readlines()
    return render_template('logs.html', logs = logs)
# Created
@general.route('/worker', methods=["GET"]) 
def worker_status():
    #worker_status_path = 'logs/worker_status.txt'
    #worker_status = None
    #with open(worker_status_path, 'r') as f:
    #    worker_status = json.load(f)
    return jsonify({'status':worker_status})