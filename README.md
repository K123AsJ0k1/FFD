# FFD

Welcome to the Federated Fraud Detection (FFD) repository, a group project created during the Networked AI Systems course 2024 of University of Helsinki. FFD is a Flask and Pytorch based federated learning application, which uses Docker compose and Oakestra for container orhestration. It provides a research simulator for local, distributed and edge enviroments with MLflow, MinIO, PostgresSQL, Prometheus, Pushgateway and Grafana integration, which can be used as a base for a custom Python Flask based federated platfrom.

## Overview of Project

- Research deployment
  - ![Central node](https://github.com/K123AsJ0k1/FFD/tree/main/code/research/central-node): The server used in federated learning
  - ![Worker node](https://github.com/K123AsJ0k1/FFD/tree/main/code/research/worker-node): The client(s) used in federated learning
  - Deployment: 
    - ![Compose](https://github.com/K123AsJ0k1/FFD/tree/main/code/research/deployment/compose): Contains deployments for demonstration and experiments 
    - ![Oakestra](https://github.com/K123AsJ0k1/FFD/tree/main/code/research/deployment/oakestra): Contains SLAs for demonstration
    - Used custom images: 
      - ![PostgreSQL](https://github.com/K123AsJ0k1/FFD/tree/main/code/research/deployment/postgreSQL)
      - ![MinIO](https://github.com/K123AsJ0k1/FFD/tree/main/code/research/deployment/minio)
      - ![MLflow](https://github.com/K123AsJ0k1/FFD/tree/main/code/research/deployment/mlflow)
      - ![Prometheus](https://github.com/K123AsJ0k1/FFD/tree/main/code/research/deployment/prometheus)
      - ![Grafana](https://github.com/K123AsJ0k1/FFD/tree/main/code/research/deployment/grafana)
- ![Demo](https://github.com/K123AsJ0k1/FFD/blob/main/notebooks/demonstrations/Research-Central-Worker-Demo.ipynb): A notebook that shows how to interact with a research FFD
- Experiments:
  - ![Notebooks](https://github.com/K123AsJ0k1/FFD/tree/main/notebooks/experiments/notebooks): Used preprocessing, scenarios and parsing for experiments
  - ![Data](https://github.com/K123AsJ0k1/FFD/tree/main/notebooks/experiments/data): Generated MinIO and MLflow data by scenarios
  - ![Images](https://github.com/K123AsJ0k1/FFD/tree/main/notebooks/experiments/images): Collected images from experiment results

## Get Started

First, clone the repository with:

```
git clone https://github.com/K123AsJ0k1/FFD.git
```

When its done, you have three deployment options of local, docker compose and oakestra with their own pros and cons:

- Local (development):
  - Pros:
    - Familiar virtual enviroment setup
    - Enables understanding of applications
    - Easy checking of logs
  - Cons:
    - Multiple workers require separate folders
    - Docker is still a requirement for integration
    - Not portable
- Docker compose (recommended):
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

From these we recommed docker compose due to ease of use and minimal steps.

### Jupyter setup

Open up a single terminal and move it to:

- FFD/code/notebooks

Create a virtual enviroment with:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start Jupyter with:

```
jupyter notebook
```

### Local Setup

Open up three terminals and move them into the following repository paths:

- FFD/code/research/central-node
- FFD/code/research/worker-node
- FFD/code/research/deployment

In the first two, create Python virtual enviroments with right packages using following:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

When these are ready, start the Jupyter notebook and open the Research-Central-Worker-Demo. Before we start the central and worker node, we first need to setup storage components using

```
docker compose -f ffd-storage-docker-compose.yaml up
```

When the logs start appearing, check out the following addresses using your browser:

- MLflow: http://127.0.0.1:5000/
- MinIo: http://127.0.0.1:9001/ 
  - User = 23034opsdjhksd
  - Password = sdkl3slömdm  

If the dashboards look normal, compose is working as intended and you can now make central and worker run. Uncomment the variables in central and worker node run.py file and run the following command:

```
python3 run.py
```

When the regular Flask logs start to show with central showing logs every 5 seconds without errors, proceed to check MinIO. When you are in MinIO, click the top option (object browser) on left, which should show central, mlflow and workers buckets. This means that the integration is succesful, which should allow you to run the demonstration notebook without problems. It is recommended to check central and worker logs using the following:

- Central: http://127.0.0.1:7500/logs
- Worker: http://127.0.0.1:7501/logs

### Docker Compose Setup

Open up three terminals and move them to to the following repository path:

- FFD/code/research/deployment/compose

Run the following commands in the first two terminals:

```
docker compose -f ffd-storage-docker-compose.yaml up

docker compose -f ffd-monitoring-docker-compose.yaml up
```

Check the following addresses:

- Grafana: http://127.0.0.1:3000/ 
  - User = admin
  - Password = admin
- MLflow: http://127.0.0.1:5000/
- MinIo: http://127.0.0.1:9001/ 
  - User = 23034opsdjhksd
  - Password = sdkl3slömdm   
- Prometheus: http://127.0.0.1:9090/
- Pushgateway: http://127.0.0.1:9091/

If there is no errors, proceed to setup the C1-W5 FFD using the third terminal:

```
docker compose -f ffd-c1-w5-nodes-docker-compose.yaml up
```

If there isn't any error logs in the terminal, check the node logs with:

- Central: http://127.0.0.1:7500/logs
- Worker-1: http://127.0.0.1:7501/logs
- Worker-2: http://127.0.0.1:7502/logs
- Worker-3: http://127.0.0.1:7503/logs
- Worker-4: http://127.0.0.1:7504/logs
- Worker-5: http://127.0.0.1:7505/logs

If these logs don't show any errors either, your FFD is now ready to run the Research-Central-Worker-Demo.

### Oakestra Setup

Please follow the Oakestra ![README](https://github.com/oakestra/oakestra) to setup a local 1-DOC setup orhestrator. When you have managed to deploy a Ngnix example, go to

- FFD/code/research/deployment/oakestra

Now, create three applications named storage, monitor and nodes in the dashboard. Then, go to the storage namespace, create a new service, select SLA and select storage-deployment.json. If service creation is succesful, deploy the services and check that all are running. To confirm that the components run, check the following:

- MLflow: http://(orhestrator_address):5000/
- MinIo: http://(orhestrator_address):9001/ 
  - User = 23034opsdjhksd
  - Password = sdkl3slömdm   

If UIs work fine, proceed to deploy monitor-deployment.json and check the UIs with

- Grafana: http://(orhestrator_address):3000/ 
  - User = admin
  - Password = admin
- Prometheus: http://(orhestrator_address):9090/

If UIs are again fine, proceed to deploy ffd-c1-w5-nodes-oakestra-deployment.json and check central logs with:

- Central: http://(orhestrator_address):7500/logs

If central is able to store workers, you know FFD is ready to run the Research-Central-Worker-Demo.

## High Level Architecture

The following image shows the components and interactions of FFD:

![architecture](https://github.com/K123AsJ0k1/FFD/blob/main/images/NAI_Report_FFD_Architecture.PNG)

**Components**
- **Central**: Creates global model, coordinates training, aggregates a new global model and evaluates the global model
- **Worker**: Creates local model and sends it to central 
- **MLflow**: Provides End-to-End ML management and model analysis tools for central and workers
- **MinIO**: Provides object storage for nodes and artifact storage for MLflow
- **PostgreSQL**: Provides metric and metadata store for MLflow
- **Prometheus**: Scrapes model, time and resource metrics stored in central and workers
- **Grafana**: Enables visualization for metrics collected by Prometheus 
