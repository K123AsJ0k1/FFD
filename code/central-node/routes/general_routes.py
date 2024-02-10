from flask import Blueprint, current_app, request, jsonify

from functions.general_functions import *
from functions.data_functions import *

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    preprocess_into_train_test_and_evaluate_tensors()
    return 'Ok', 200

@general.route('/update', methods=["POST"]) 
def model_update():
    print('Model update')
    #sent_payload = request.json
    return 'Ok', 200
