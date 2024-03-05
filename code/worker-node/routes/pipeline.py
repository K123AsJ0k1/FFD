from flask import Blueprint, request, jsonify
import json

from functions.storage import store_training_context

pipeline = Blueprint('pipeline', __name__)

# Refactored
@pipeline.route('/context', methods=["POST"]) 
def set_training_context():
    sent_payload = json.loads(request.json)
     
    sent_parameters = sent_payload['parameters']
    sent_global_model = sent_payload['global-model']
    sent_worker_data_list = sent_payload['worker-data-list']
    sent_worker_data_columns = sent_payload['worker-data-columns']
    
    status = store_training_context(
        parameters = sent_parameters ,
        global_model = sent_global_model,
        df_data = sent_worker_data_list,
        df_columns = sent_worker_data_columns
    )
    
    return 'Ok', 200