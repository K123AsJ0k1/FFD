from flask import Flask, request
from config import Config
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import os

def create_app():
    app = Flask(__name__)

    os.makedirs('logs', exist_ok=True)
    os.makedirs('parameters', exist_ok=True)
    os.makedirs('status', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('resources', exist_ok=True)
    os.makedirs('tensors', exist_ok=True)

    worker_log_path = 'logs/worker.log'
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
    
    from functions.storage_functions import initilize_storage_templates
    initilize_storage_templates()
    
    #status = initilize_worker_status()
    #app.logger.info('Worker status created: ' + str(status))
    
    scheduler = BackgroundScheduler(daemon = True)
    from functions.fed_functions import update_pipeline
    #from functions.fed_functions import send_status_to_central
    #from functions.fed_functions import worker_federated_pipeline
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

    from routes.general_routes import general
    app.logger.info('Routes imported')

    app.register_blueprint(general)
    app.logger.info('Routes registered')
    
    app.logger.info('Worker ready')
    os.environ['STATUS'] = 'waiting'
    return app