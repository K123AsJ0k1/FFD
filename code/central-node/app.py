from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler

import threading
import logging
import os
# Refactor
def create_app():
    app = Flask(__name__)

    app.file_lock = threading.Lock()

    os.makedirs('storage', exist_ok=True)
    os.makedirs('storage/logs', exist_ok=True)
    os.makedirs('storage/parameters', exist_ok=True)
    os.makedirs('storage/status', exist_ok=True)
    os.makedirs('storage/data', exist_ok=True)
    os.makedirs('storage/models', exist_ok=True)
    os.makedirs('storage/metrics', exist_ok=True)
    os.makedirs('storage/resources', exist_ok=True)
    os.makedirs('storage/tensors', exist_ok=True)
    
    central_log_path = 'storage/logs/central.log'
    if os.path.exists(central_log_path):
        os.remove(central_log_path)

    logger = logging.getLogger('central-logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(central_log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    app.logger = logger

    from functions.initilization import initilize_storage_templates
    initilize_storage_templates(
        file_lock = app.file_lock
    )

    scheduler = BackgroundScheduler(daemon = True)
    from functions.pipeline import data_pipeline
    from functions.pipeline import model_pipeline
    #from functions.pipeline import update_pipeline
    #from functions.pipeline import aggregation_pipeline
    
    given_args = [
        app.file_lock,
        app.logger
    ] 
    # Works 40 sec
    scheduler.add_job(
        func = data_pipeline,
        trigger = "interval",
        seconds = 40,
        args = given_args 
    )
    # Works 60 sec
    scheduler.add_job(
        func = model_pipeline,
        trigger = "interval",
        seconds = 60,
        args = given_args 
    )
    # Works
    #scheduler.add_job(
    #    func = update_pipeline,
    #    trigger = "interval",
    #    seconds = 20,
    #    args = given_args 
    # )
    # Works
    #scheduler.add_job(
    #    func = aggregation_pipeline,
    #    trigger = "interval",
    #    seconds = 80,
    #    args = given_args 
    #)
    scheduler.start()
    app.logger.info('Scheduler ready')

    from routes.general import general
    from routes.model import model
    from routes.orchestration import orchestration
    from routes.pipeline import pipeline
    app.logger.info('Routes imported')

    app.register_blueprint(general)
    app.register_blueprint(model)
    app.register_blueprint(orchestration)
    app.register_blueprint(pipeline)
    app.logger.info('Routes registered')
    
    app.logger.info('Central ready')
    os.environ['STATUS'] = 'waiting'
    return app