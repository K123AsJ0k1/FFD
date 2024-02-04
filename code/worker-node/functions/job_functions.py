from flask import current_app
import requests

def send_update():
    print('Send update')
    address = 'http://127.0.0.1:7600/update'
    try:
        response = requests.post(
            url = address
        )
        print(response.status_code)
    except Exception as e:
        current_app.logger.error('Update sending error')
        current_app.logger.error(e)
