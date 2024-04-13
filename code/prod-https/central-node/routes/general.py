from flask import Blueprint, render_template, current_app
from functions.general import get_central_logs
from prometheus_client import generate_latest

#from functions.management.objects import get_folder_object_paths
#from functions.processing.split import get_data_workers

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    return 'Ok', 200
# Created and works
@general.route('/metrics') 
def metrics():
    return generate_latest(current_app.prometheus_registry)
# Refactored and works
@general.route('/logs', methods=["GET"]) 
def central_logs():
    current_logs = get_central_logs()
    return render_template('logs.html', logs = current_logs)