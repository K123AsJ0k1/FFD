# FFD

Welcome to the Federated Fraud Detection (FFD) repository, a group project created during the Networked AI Systems course 2024 of University of Helsinki. FFD is a Flask and Pytorch based federated learning application, which uses Docker compose and Oakestra for container orhestration. It provides a research simulator for local, distributed and edge enviroments with MLflow, MinIO, PostgresSQL, Prometheus, Pushgateway and Grafana integration, which can be used as a base for a custom Python Flask based federated platfrom.

## Overview of Project

## Get Started

First, clone the repository with:

```
git clone https://github.com/K123AsJ0k1/FFD.git
```

When its done, you have three deployment options of local, docker compose and oakestra with their own pros and cons:

- Local:
  - Pros:
    - Familiar virtual enviroment setup
    - Enables understanding of applications
    - Easy checking of logs
  - Cons:
    - Multiple workers require separate folders
    - Docker is still a requirement for integration
    - Not portable
- Docker compose:
  - Pros:
    - Straightforward if Docker is properly installed and configured
    - Enables proper enviroment with multiple workers
    - Easy orhestration debugging
  - Cons:
    - Distributed networking requires custom configuration
    - Requires enough resources to run integration components
    - Kubernetes with Kind is better option for production
- Oakestra:
  - Pros:
    - A proper orhestration for edge enviroments
    - Pretty straightforward setup of root and workers
    - Quite easy setup with existing SLAs
  - Cons:
    - Oakestra is not currently meant for production
    - Requires a longer setup than other options
    - Much smaller community than Kuberentes

We recommend trying all options to see what fits best, but the target enviroment for FFD is Oakestra.

### Local Setup

Open up four terminals and move them into the following repository paths:

- FFD/code/notebooks/demonstrations/Production-Central-Worker
- FFD/code/prod-https/central-node
- FFD/code/prod-https/worker-node
- FFD/code/prod-https/deployment

In the first three, create Python virtual enviroments with right packages using following:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

When these are ready, open up the Jupyter notebook with

```
jupyter notebook
```

and open the Production-Central-Worker-Demonstration. Before we start the central and worker node, we first need to use the development-docker-compose.yaml via the following command:

```
docker compose -f development-docker-compose.yaml up
```

When the logs start appearing, check out the following addresses using your browser:

- Grafana: http://127.0.0.1:3000/ 
  - User = admin
  - Password = admin
- MLflow: http://127.0.0.1:5000/
- MinIo: http://127.0.0.1:9001/ 
  - User = minio
  - User = minio123   
- Prometheus: http://127.0.0.1:9090/
- Pushgateway: http://127.0.0.1:9091/

If the dashboards look normal, compose is working as intended and you can now make central and worker run using the following command:

```
python3 run.py
```

When the regular Flask logs start to show with central showing logs every 5 seconds without errors, proceed to check MinIO. When you are in MinIO, click the top option (object browser) on left, which should show central, mlflow and workers buckets. This means that the integration is succesful, which should allow you to run the demonstration notebook without problems. 

### Docker Compose Setup

Open up one terminal and move it to the following repository path:

- FFD/code/prod-https/deployment

Run the following command to make the stack run:

```
docker compose -f development-docker-compose.yaml up
```

Check the following addresses:

- Grafana: http://127.0.0.1:3000/ 
  - User = admin
  - Password = admin
- MLflow: http://127.0.0.1:5000/
- Central: http://127.0.0.1:7500/logs
- Worker-1: http://127.0.0.1:7501/logs
- Worker-2: http://127.0.0.1:7502/logs
- Worker-3: http://127.0.0.1:7503/logs
- Worker-4: http://127.0.0.1:7504/logs
- Worker-5: http://127.0.0.1:7505/logs
- MinIo: http://127.0.0.1:9001/ 
  - User = minio
  - User = minio123   
- Prometheus: http://127.0.0.1:9090/
- Pushgateway: http://127.0.0.1:9091/

If there is no errors, proceed to run the demonstration notebook.

### Oakestra Setup