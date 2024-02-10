from flask import Blueprint, current_app, request, jsonify

from functions.general_functions import *
from functions.data_functions import *
from functions.model_functions import *

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    initial_model_training()
    return 'Ok', 200
# Works
@general.route('/register', methods=["POST"])
def worker_registration():
    received_worker_ip = request.remote_addr
    store_worker_ip(
        worker_ip = received_worker_ip
    )
    return 'Ok', 200  

@general.route('/update', methods=["POST"]) 
def model_update():
    print('Model update')
    #sent_payload = request.json
    return 'Ok', 200
