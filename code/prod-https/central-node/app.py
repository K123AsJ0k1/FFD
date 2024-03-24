from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from minio import Minio
from prometheus_client import CollectorRegistry
from mlflow import MlflowClient

import threading
import logging
import os
# Refactored and works
def create_app():
    app = Flask(__name__)

    app.file_lock = threading.Lock()

    os.makedirs('logs', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)

    central_log_path = 'logs/central.log'
    if os.path.exists(central_log_path):
        os.remove(central_log_path)

    logger = logging.getLogger('central-logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(central_log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    app.logger = logger
    
    minio_client = Minio(
        endpoint = "127.0.0.1:9000", 
        access_key = 'minio', 
        secret_key = 'minio123',
        secure = False
    )
    app.minio_client = minio_client
    app.logger.warning('Minion client ready')

    mlflow_cient = MlflowClient(
        tracking_uri = "http://127.0.0.1:5000"
    )
    app.mlflow_client = mlflow_cient
    app.logger.warning('MLflow client ready')

    registry = CollectorRegistry()
    app.prometheus_registry = registry
    app.prometheus_metrics = {
        'central-global': None,
        'central-global-names': None,
        'central-resources': None,
        'central-resources-names': None
    }
    app.logger.warning('Prometheus registry and metrics ready')

    from functions.initilization import initilize_minio, initilize_prometheus_gauges
    initilize_minio(
        logger = app.logger,
        minio_client = minio_client
    )
    initilize_prometheus_gauges(
        prometheus_registry = app.prometheus_registry,
        prometheus_metrics = app.prometheus_metrics
    )

    scheduler = BackgroundScheduler(daemon = True)
    from functions.management.pipeline import processing_pipeline
    from functions.management.pipeline import model_pipeline
    #from functions.pipeline import update_pipeline
    #from functions.pipeline import aggregation_pipeline
    
    given_args = [
        app.file_lock,
        app.logger,
        app.minio_client,
        app.prometheus_registry,
        app.prometheus_metrics
    ] 
    # Works 30 sec
    scheduler.add_job(
        func = processing_pipeline,
        trigger = "interval",
        seconds = 30,
        args = given_args 
    )
    given_args = [
        app.file_lock,
        app.logger,
        app.minio_client,
        app.mlflow_client,
        app.prometheus_registry,
        app.prometheus_metrics,
    ] 
    # Works 60 sec
    scheduler.add_job(
        func = model_pipeline,
        trigger = "interval",
        seconds = 60,
        args = given_args 
    )
    # Works 20 sec
    #scheduler.add_job(
    #    func = update_pipeline,
    #    trigger = "interval",
    #    seconds = 20,
    #    args = given_args 
    #)
    # Works 40 sec
    #scheduler.add_job(
    #    func = aggregation_pipeline,
    #    trigger = "interval",
    #    seconds = 40,
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