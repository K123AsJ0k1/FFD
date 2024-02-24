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
    from functions.storage_functions import initilize_worker_status
    status = initilize_worker_status()
    app.logger.warning('Worker status created: ' + str(status))
    
    scheduler = BackgroundScheduler(daemon = True)
    from functions.fed_functions import send_status_to_central
    from functions.fed_functions import worker_federated_pipeline
    given_args = [app.logger,app.config['CENTRAL_ADDRESS']]
    scheduler.add_job(
        func = send_status_to_central,
        trigger = "interval",
        seconds = 5,
        args = given_args
    )
    given_args = [app.logger,app.config['CENTRAL_ADDRESS']]
    scheduler.add_job(
        func = worker_federated_pipeline,
        trigger = "interval",
        seconds = 50,
        args = given_args 
    )
    scheduler.start()
    app.logger.warning('Scheduler ready')

    from routes.general_routes import general
    app.logger.warning('Routes imported')

    app.register_blueprint(general)
    app.logger.warning('Routes registered')
    
    app.logger.warning('Worker ready')
    return app