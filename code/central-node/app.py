from flask import Flask
from config import Config
from apscheduler.schedulers.background import BackgroundScheduler

import logging
import os
# Needs refactoring
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
    from functions.storage_functions import initilize_training_status
    status = initilize_training_status()
    app.logger.warning('Training status created: ' + str(status))
    
    scheduler = BackgroundScheduler(daemon = True)
    from functions.fed_functions import send_context_to_workers
    from functions.fed_functions import central_federated_pipeline
    given_args = [
        app.logger, 
        app.config['GLOBAL_PARAMETERS'], 
        app.config['CENTRAL_PARAMETERS'], 
        app.config['WORKER_PARAMETERS']
    ] 
    scheduler.add_job(
        func = send_context_to_workers,
        trigger = "interval",
        seconds = 30,
        args = given_args 
    )
    given_args = [
        app.logger, 
        app.config['GLOBAL_PARAMETERS'], 
        app.config['CENTRAL_PARAMETERS']
    ]
    scheduler.add_job(
        func = central_federated_pipeline,
        trigger = "interval",
        seconds = 60,
        args = given_args 
    )
    scheduler.start()
    app.logger.warning('Scheduler ready')

    from routes.general_routes import general
    app.logger.warning('Routes imported')

    app.register_blueprint(general)
    app.logger.warning('Routes registered')
    
    app.logger.warning('Central ready')
    os.environ['STATUS'] = 'waiting'
    return app