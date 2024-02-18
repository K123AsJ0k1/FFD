from flask import Flask
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

    os.environ['STATUS'] = 'waiting'

    scheduler = BackgroundScheduler(daemon = True)
    from functions.data_functions import send_context_to_workers
    send_update_args = [app.logger]
    scheduler.add_job(
        func = send_context_to_workers,
        trigger = "interval",
        seconds = 20,
        args = send_update_args
    )
    scheduler.start()
    app.logger.warning('Scheduler ready')

    from routes.general_routes import general
    app.logger.warning('Routes imported')

    app.register_blueprint(general)
    app.logger.warning('Routes registered')
    
    app.logger.warning('Node ready')
    return app