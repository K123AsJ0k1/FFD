import requests
# works, when docker run --network=host -p 7500:7500 worker:ping
def send_update(logger, central_address):
    logger.warning('Send update')
    address = central_address + '/update'
    try:
        response = requests.post(
            url = address
        )
        logger.warning(response.status_code)
    except Exception as e:
        logger.error('Update sending error')
        logger.error(e) 
        
