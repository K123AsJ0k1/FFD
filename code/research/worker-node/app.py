from flask import Flask

import json
import threading
import logging
import os
import uuid

from apscheduler.schedulers.background import BackgroundScheduler
from minio import Minio
from prometheus_client import CollectorRegistry
from mlflow import MlflowClient

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
        os.environ['WORKER_ADDRESS'] = critical_variables['worker-address']
        os.environ['WORKER_SYSTEM_MONITOR'] = critical_variables['worker-system-monitor']
    else:
        critical_variables = {
            'worker-id': str(uuid.uuid4()),
            'central-address': os.environ.get('CENTRAL_ADDRESS'),
            'central-port': os.environ.get('CENTRAL_PORT'),
            'worker-port': os.environ.get('WORKER_PORT'),
            'worker-address': os.environ.get('WORKER_ADDRESS'),
            'worker-system-monitor': os.environ.get('WORKER_SYSTEM_MONITOR')
        }
        with open(critical_path, 'w') as f:
            json.dump(critical_variables, f, indent=4)
        os.environ['WORKER_ID'] = critical_variables['worker-id']
    
    logger = logging.getLogger('worker-logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(worker_log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    app.logger = logger

    minio_client = Minio(
        endpoint = os.environ.get('MINIO_ENDPOINT'), 
        access_key = os.environ.get('AWS_ACCESS_KEY_ID'), 
        secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY'),
        secure = False
    )
    app.minio_client = minio_client
    app.logger.info('Minion client ready')

    mlflow_client = MlflowClient(
        tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    )
    app.mlflow_client = mlflow_client
    app.logger.info('MLflow client ready')

    registry = CollectorRegistry()
    app.prometheus_registry = registry
    app.prometheus_metrics = {
        'central-global': None,
        'central-global-names': None,
        'central-resources': None,
        'central-resources-names': None
    }
    app.logger.info('Prometheus registry and gauges ready')

    from functions.initilization import initilize_envs, initilize_minio, initilize_prometheus_gauges
    initilize_envs(
        file_lock = app.file_lock,
        logger = app.logger,
        minio_client = minio_client
    )
    success = False
    for tries in range(0,10):
        if success:
            break
        success = initilize_minio(
            file_lock = app.file_lock,
            logger = app.logger,
            minio_client = minio_client
        )
    initilize_prometheus_gauges(
        prometheus_registry = app.prometheus_registry,
        prometheus_metrics = app.prometheus_metrics
    )
    
    scheduler = BackgroundScheduler(daemon = True)
    from functions.management.pipeline import system_monitoring, server_monitoring, status_pipeline, data_pipeline, model_pipeline, update_pipeline
    
    given_args = [
        app.file_lock,
        app.logger,
        app.minio_client,
        app.prometheus_registry,
        app.prometheus_metrics
    ]

    # Works 5 sec
    scheduler.add_job(
        func = server_monitoring,
        trigger = "interval",
        seconds = 5,
        args = given_args 
    )
    if os.environ.get('WORKER_SYSTEM_MONITOR') == '1':
        # Works 10 sec
        scheduler.add_job(
            func = system_monitoring,
            trigger = "interval",
            seconds = 10,
            args = given_args 
        )
    
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
    app.logger.info('Routes imported')

    app.register_blueprint(general)
    app.register_blueprint(model)
    app.register_blueprint(orchestration)
    app.logger.info('Routes registered')
    
    app.logger.info('Worker ready')
    return app