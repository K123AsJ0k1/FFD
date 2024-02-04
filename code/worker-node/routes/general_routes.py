from flask import Blueprint, current_app, request, jsonify

from functions.general_functions import *

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    print(get_debug_mode())
    return 'Ok', 200

#@general.before_app_request
#def start_scheduler():
#    print('Scheduler start')
#    SCHEDULER = current_app.scheduler
#    SCHEDULER.start()
#    SCHEDULER.add_job(
#        func = send_update,
#        trigger = "interval",
#        seconds = 30
#    )