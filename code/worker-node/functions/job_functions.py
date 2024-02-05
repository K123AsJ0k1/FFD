import requests

def send_update(central_address):
    print('Send update')
    address = central_address + '/update'
    try:
        response = requests.post(
            url = address
        )
        print(response.status_code)
    except Exception as e:
        print('Update sending error')
        print(e)
