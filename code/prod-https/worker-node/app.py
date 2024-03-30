from flask import Flask, request
from apscheduler.schedulers.background import BackgroundScheduler
from minio import Minio
from prometheus_client import CollectorRegistry
from mlflow import MlflowClient
import json

import threading
import logging
import os
import uuid

def create_app():
    app = Flask(__name__)
    app.file_lock = threading.Lock()

    os.makedirs('logs', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
   
    worker_log_path = 'logs/worker.log'
    if os.path.exists(worker_log_path):
        os.remove(worker_log_path)

    critical_path = 'logs/critical.txt'
    if os.path.exists(critical_path):
        critical_variables = None
        with open(critical_path, 'r') as f:
            critical_variables = json.load(f)
        os.environ['WORKER_ID'] = critical_variables['worker-id']
        os.environ['CENTRAL_ADDRESS'] = critical_variables['central-address']
        os.environ['CENTRAL_PORT'] = critical_variables['central-port']
        os.environ['WORKER_PORT'] = critical_variables['worker-port']
    else:
        # Refactor to handle given envs
        critical_variables = {
            'worker-id': str(uuid.uuid4()),
            'central-address': '127.0.0.1',
            'central-port': '7600',
            'worker-port': '7500'
        }
        with open(critical_path, 'w') as f:
            json.dump(critical_variables, f, indent=4)
        os.environ['WORKER_ID'] = critical_variables['worker-id']
        os.environ['CENTRAL_ADDRESS'] = critical_variables['central-address']
        os.environ['CENTRAL_PORT'] = critical_variables['central-port']
        os.environ['WORKER_PORT'] = critical_variables['worker-port']
    
    logger = logging.getLogger('worker-logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(worker_log_path)
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

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
    os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
    mlflow_client = MlflowClient(
        tracking_uri = "http://127.0.0.1:5000"
    )
    app.mlflow_client = mlflow_client
    app.logger.warning('MLflow client ready')

    registry = CollectorRegistry()
    app.prometheus_registry = registry
    app.prometheus_metrics = {
        'central-global': None,
        'central-global-names': None,
        'central-resources': None,
        'central-resources-names': None
    }
    app.logger.warning('Prometheus registry and gauges ready')

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
    from functions.management.pipeline import status_pipeline, data_pipeline, model_pipeline, update_pipeline
    
    given_args = [
        app.file_lock,
        app.logger,
        app.minio_client,
        app.prometheus_registry,
        app.prometheus_metrics
    ]
    scheduler.add_job(
        func = status_pipeline,
        trigger = "interval",
        seconds = 10,
        args = given_args
    )
    scheduler.add_job(
        func = data_pipeline,
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
        app.prometheus_metrics
    ]
    scheduler.add_job(
        func = model_pipeline,
        trigger = "interval",
        seconds = 60,
        args = given_args
    )
    given_args = [
        app.file_lock,
        app.logger,
        app.minio_client,
        app.prometheus_registry,
        app.prometheus_metrics
    ]
    scheduler.add_job(
        func = update_pipeline,
        trigger = "interval",
        seconds = 40,
        args = given_args
    )
    
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