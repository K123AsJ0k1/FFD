# Containerized Flask Application Development

For all Flask applications with the exception of really simple ones, one of the best development patterns is the [application factory](https://dev.to/bredmond1019/flask-application-factory-1j81) method, where we aim to create small and modular files that are capable of dynamic configuration and replication depending on the enviroment. The main parts of this pattern are:

- Application factory like app.py:
  - Uses configuration and routes to define the application 
- Configuration file like config.py
  - Gives predefined enviromental variables for different situations 
- Blueprints like general_routes.py
  - Defines the available actions the application can take 
- Functions like general_functions.py
  - Defines the functions used by blueprints 
- Runner like run.py
  - Starts up the application with a wanted configuration

This guide shows the basics for setting up and developing these parts.

## Setup 

To start, we need to create the simple templates, enviroment and testing to start working. For templates there are different options found from guides like [1](https://flask.palletsprojects.com/en/2.3.x/patterns/appfactories/) and [2](https://dev.to/bredmond1019/flask-application-factory-1j81), but the one I have found useful for Kubernetes deployments is the one we can see in the worker-node folder. It consists of these files:

### run.py
  
```
from app import create_app
import warnings

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    app = create_app()
    app.run(host = '0.0.0.0', port = 7500)
```

### config.py

```
class Config:
    DEBUG = None
    
class DevConfig(Config):
    DEBUG = True

class ProdConfig(Config):
    DEBUG = False
```

### app.py

```
from flask import Flask
from config import Config
import logging

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

    from routes.general_routes import general
    app.logger.warning('Routes imported')

    app.register_blueprint(general)
    app.logger.warning('Routes registered')
    
    app.logger.warning('Node ready')
    return app
```

### Dockerfile

```
# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

WORKDIR /worker

COPY config.py .

COPY run.py .

COPY app.py .

COPY routes /worker/routes

COPY functions /worker/functions

COPY requirements.txt .

RUN pip3 install -r requirements.txt

EXPOSE 7500

CMD ["python", "run.py"]
```

### routes/general_routes.py

```
from flask import Blueprint, current_app, request, jsonify

from functions.general_functions import *

general = Blueprint('general', __name__)

@general.route('/demo', methods=["GET"]) 
def demo():
    print(get_debug_mode())
    return 'Ok', 200

```

### functions/general_functions.py

```
from flask import current_app
from typing import Dict

def get_debug_mode() -> list:
   return current_app.config['DEBUG']
```

In order to run this application, we need to use Python virtual enviroment, which we can setup with:

```
cd worker-node
python3 -m venv venv
source venv/bin/activate # Linux
venv\scripts\activate # Windows
pip install Flask
python3 run.py
```

In order to test that its demo route works, I recommend using Jupyter notebooks to sent HTTP requests, which we can do the following way

```
cd notebooks
python3 -m venv venv
source venv/bin/activate # Linux
venv\scripts\activate # Windows
pip install notebooks
jupyter notebook
go to testing folder and run the blocks
```

We now have all of the necessery blocks to start the development of our application. To ensure progress, it is recommended to use Git and GitHub to its fullest by using [branches](https://www.atlassian.com/git/tutorials/using-branches) and [pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request). However, in my opinion the easiest approach is first testing different functions and components in notebooks and then bring them to the flask application. Thus, here are some important commands:

```
CTRL + C # Stopping applications

git add . # Put all changes to staging

git commit -m "Message" # Commit changes

git push # Push changes to repo

git branch --list # Lists all branches

git branch (branch name) # Creates a branch without checkout

git branch -d (branch name) # safe delete a branch

git branch -D (branch name) # Force felete a branch

git branch -m (new branch name) # Rename current branch
```

## Workflow

In general, the workflow for adding a new feature into the application is this:

1. Check the current state of the application
2. Evaluate if there is any practical overlap with existing code, which you could use to reduce work
3. Analyze how much new would there be in practice by creating a function and running it either in notebook or in the demo route
4. When the function works as expected, check if it provides the wanted feature. If not, jump back to step 3.
5. When the function(s) is/are ready, create a route, which utilizes it/them and test it either with HTTP requests or application restarts
6. After all the necessery manual tests are done, remember to update requirements.txt and Dockerfile to build the new image

```
pip freeze >> requirements.txt # Creates a requirements file. Old one should be deleted before running this

docker build -t image_name:image_version . # Creates a image called image_name:image_version

docker images # Shows current images

docker run image_name:image_version # Creates a container using a given image
```

---





