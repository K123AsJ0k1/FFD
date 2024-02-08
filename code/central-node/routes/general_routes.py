from flask import Blueprint, current_app, request, jsonify

from functions.general_functions import *

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    print(get_debug_mode())
    return 'Ok', 200

@general.route('/update', methods=["POST"]) 
def model_update():
    print('Model update')
    #sent_payload = request.json
    return 'Ok', 200