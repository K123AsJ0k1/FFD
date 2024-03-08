from flask import Blueprint, request, jsonify
import json

from functions.storage import store_worker

orchestration = Blueprint('orchestration', __name__)
# Refactored and works
@orchestration.route('/status', methods=["POST"])
def worker_status():
    received_worker_ip = request.remote_addr
    sent_payload = json.loads(request.json)
    sent_status = sent_payload['status']
    sent_metrics = sent_payload['metrics']

    payload = store_worker(
        address = received_worker_ip,
        status = sent_status,
        metrics = sent_metrics
    )

    return jsonify(payload) 