from flask import Blueprint, request, jsonify,current_app
import json
from functions.management.storage import store_worker

orchestration = Blueprint('orchestration', __name__)
# Refactored and works
@orchestration.route('/status', methods=["POST"])
def worker_status():
    received_worker_ip = request.remote_addr
    sent_payload = json.loads(request.json)
    sent_status = sent_payload['status']

    payload = store_worker(
        logger = current_app.logger,
        minio_client = current_app.minio_client,
        prometheus_registry = current_app.prometheus_registry,
        prometheus_metrics = current_app.prometheus_metrics,
        address = received_worker_ip,
        status = sent_status
    )

    return jsonify(payload) 