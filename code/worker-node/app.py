from flask import Flask, request
from config import Config
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import os

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

    worker_log_path = 'storage/logs/worker.log'
    if os.path.exists(worker_log_path):
        os.remove(worker_log_path)

    app.config.from_object(Config)
    logger = logging.getLogger('worker-logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(worker_log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    app.logger = logger
    
    from functions.initilization import initilize_storage_templates
    initilize_storage_templates()
    
    scheduler = BackgroundScheduler(daemon = True)
    from functions.pipeline import update_pipeline
    given_args = [app.logger]
    scheduler.add_job(
        func = update_pipeline,
        trigger = "interval",
        seconds = 5,
        args = given_args
    )

    #given_args = [app.logger,app.config['CENTRAL_ADDRESS']]
    #scheduler.add_job(
    #    func = send_status_to_central,
    #    trigger = "interval",
    #    seconds = 5,
    #    args = given_args
    #)
    #given_args = [app.logger,app.config['CENTRAL_ADDRESS']]
    #scheduler.add_job(
    #    func = worker_federated_pipeline,
    #    trigger = "interval",
    #    seconds = 50,
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
    
    app.logger.info('Worker ready')
    os.environ['STATUS'] = 'waiting'
    return app