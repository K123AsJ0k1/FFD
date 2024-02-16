from flask import Flask, request
from config import Config
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import os

def create_app():
    app = Flask(__name__)

    app.config.from_object(Config)
    logging.basicConfig(level = logging.WARNING, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    enviroment = 'PROD'
    if enviroment == 'DEV':
        logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        app.logger.warning('Choosen enviroment is development')
        app.config.from_object('config.DevConfig')
    elif enviroment == 'PROD':
        app.logger.warning('Choosen enviroment is production')
        app.config.from_object('config.ProdConfig')

    os.environ['STATUS'] = 'waiting'

    scheduler = BackgroundScheduler(daemon = True)
    from functions.data_functions import send_status_to_central
    from functions.model_functions import send_update
    send_update_args = [app.logger,app.config['CENTRAL_ADDRESS']]
    scheduler.add_job(
        func = send_status_to_central,
        trigger = "interval",
        seconds = 5,
        args = send_update_args
    )
    scheduler.add_job(
        func = send_update,
        trigger = "interval",
        seconds = 5,
        args = send_update_args
    )
    scheduler.start()
    app.logger.warning('Scheduler ready')

    from routes.general_routes import general
    app.logger.warning('Routes imported')

    app.register_blueprint(general)
    app.logger.warning('Routes registered')
    
    app.logger.warning('Node ready')
    return app