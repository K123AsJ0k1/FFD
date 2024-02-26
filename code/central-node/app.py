from flask import Flask
from config import Config
from apscheduler.schedulers.background import BackgroundScheduler

import logging
import os
# Refactored and works
def create_app():
    app = Flask(__name__)
    
    central_log_path = 'logs/central.log'
    if os.path.exists(central_log_path):
        os.remove(central_log_path)

    app.config.from_object(Config)
    logger = logging.getLogger('central-logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(central_log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    app.logger = logger

    from functions.storage_functions import initilize_training_status
    status = initilize_training_status()
    app.logger.info('Training status created: ' + str(status))
    
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
        app.config['CENTRAL_PARAMETERS'],
        app.config['WORKER_PARAMETERS']
    ]
    scheduler.add_job(
        func = central_federated_pipeline,
        trigger = "interval",
        seconds = 60,
        args = given_args 
    )
    scheduler.start()
    app.logger.info('Scheduler ready')

    from routes.general_routes import general
    app.logger.info('Routes imported')

    app.register_blueprint(general)
    app.logger.info('Routes registered')
    
    app.logger.info('Central ready')
    os.environ['STATUS'] = 'waiting'
    return app