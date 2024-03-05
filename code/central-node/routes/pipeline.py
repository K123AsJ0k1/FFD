from flask import Blueprint, request, jsonify
import json

from functions.storage import store_training_context, store_update
from functions.pipeline import start_pipeline

pipeline = Blueprint('pipeline', __name__)
# Refactored and works
@pipeline.route('/context', methods=["POST"]) 
def set_training_context():
    sent_payload = json.loads(request.json)

    sent_parameters = sent_payload['parameters']
    sent_data = sent_payload['data']
    sent_columns = sent_payload['columns']

    status = store_training_context(
        parameters = sent_parameters,
        df_data = sent_data,
        df_columns = sent_columns
    )

    return jsonify({'stored': status})
# Refactored and works
@pipeline.route('/start', methods=["POST"])
def start_training():
    status = start_pipeline()
    return 'Ok', 200
# Refactor
@pipeline.route('/update', methods=["POST"]) 
def set_worker_update(): 
    sent_payload = json.loads(request.json)
    
    sent_worker_id = sent_payload['worker-id']
    sent_local_model = sent_payload['local-model']
    sent_cycle = sent_payload['cycle']
    sent_train_size = sent_payload['train-size']

    store_update(
        worker_id = sent_worker_id,
        local_model = sent_local_model,
        cycle = sent_cycle,
        train_size = sent_train_size 
    ) 
    
    return 'Ok', 200