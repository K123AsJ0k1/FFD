from flask import current_app,request
import requests

def get_debug_mode() -> list:
   return current_app.config['DEBUG'] 

def register_worker(logger, central_address):
    logger.warning('Register worker')
    address = central_address + '/register'
    try:
        response = requests.post(
            url = address
        )
        logger.warning(response.status_code)
    except Exception as e:
        logger.error('Registration error')
        logger.error(e) 