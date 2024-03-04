from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler

import logging
import os
# Refactor
def create_app():
    app = Flask(__name__)

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

    from functions.initilization_functions import initilize_storage_templates
    initilize_storage_templates()
    #app.logger.info('Training status created: ' + str(status))
    
    scheduler = BackgroundScheduler(daemon = True)
    #from functions.fed_functions import send_context_to_workers
    from functions.pipeline_functions import data_pipeline
    from functions.pipeline_functions import model_pipeline
    from functions.pipeline_functions import update_pipeline
    #from functions.fed_functions import aggregation_pipeline
    given_args = [
        app.logger
    ] 
    scheduler.add_job(
        func = data_pipeline,
        trigger = "interval",
        seconds = 20,
        args = given_args 
    )
    #scheduler.add_job(
    #    func = model_pipeline,
    #    trigger = "interval",
    #    seconds = 120,
    #    args = given_args 
    #)
    #scheduler.add_job(
    #    func = update_pipeline,
    #    trigger = "interval",
    #    seconds = 20,
    #    args = given_args 
    #)
    
    scheduler.start()
    app.logger.info('Scheduler ready')

    from routes.general_routes import general
    from routes.model_routes import model
    from routes.orchestration_routes import orchestration
    from routes.pipeline_routes import pipeline
    app.logger.info('Routes imported')

    app.register_blueprint(general)
    app.register_blueprint(model)
    app.register_blueprint(orchestration)
    app.register_blueprint(pipeline)
    app.logger.info('Routes registered')
    
    app.logger.info('Central ready')
    os.environ['STATUS'] = 'waiting'
    return app