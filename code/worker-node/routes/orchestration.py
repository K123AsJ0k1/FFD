from flask import Blueprint, current_app, request, jsonify, render_template
import json

from functions.storage import store_central_address

orchestration = Blueprint('orchestration', __name__)
# Created and works
@orchestration.route('/point', methods=["POST"]) 
def set_point_to_central():
    sent_payload = json.loads(request.json)
    
    sent_central_address = sent_payload['central-address']

    status = store_central_address(
        central_address = sent_central_address
    )

    return 'Ok', 200