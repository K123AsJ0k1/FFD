# FFD

Welcome to the Federated Fraud Detection (FFD) repository, a group project created during the Networked AI Systems course 2024 of University of Helsinki. FFD is a Flask and Pytorch based federated learning application, which uses Docker compose and Oakestra for container orhestration. It provides a research simulator for local, distributed and edge enviroments with MLflow, MinIO, PostgresSQL, Prometheus, Pushgateway and Grafana integration, which can be used as a base for a custom Python Flask based federated platfrom.

## Overview of Project

- Research deployment
  - ![Central node](): The server used in federated learning
  - ![Worker node](): The client(s) used in federated learning
  - Deployment: 
    - ![Compose](): Contains deployments for demonstration and experiment 
    - ![Oakestra](): Contains SLAs for demonstration
    - Used custom images: 
      - ![PostgreSQL]()
      - ![MinIO]()
      - ![MLflow]()
      - ![Prometheus]()
      - ![Grafana]()
- ![Demonstration](): A notebook that shows how to interact with a research FFD
- Experiments
  - Notebooks
  - Data
  - Images

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

We recommend trying all options to see what fits best, but we recommend docker compose.

### Local Setup

Open up four terminals and move them into the following repository paths:

- FFD/code/notebooks
- FFD/code/research/central-node
- FFD/code/research/worker-node
- FFD/code/research/deployment

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

and open the Research-Central-Worker-Demo. Before we start the central and worker node, we first need to setup storage components using

```
docker compose -f ffd-storage-docker-compose.yaml up
```

When the logs start appearing, check out the following addresses using your browser:

- Grafana: http://127.0.0.1:3000/ 
  - User = admin
  - Password = admin
- MLflow: http://127.0.0.1:5000/
- MinIo: http://127.0.0.1:9001/ 
  - User = 23034opsdjhksd
  - Password = sdkl3slömdm  
- Prometheus: http://127.0.0.1:9090/
- Pushgateway: http://127.0.0.1:9091/

If the dashboards look normal, compose is working as intended and you can now make central and worker run. Uncomment the variables in central and worker node run.py file and run the following command:

```
python3 run.py
```

When the regular Flask logs start to show with central showing logs every 5 seconds without errors, proceed to check MinIO. When you are in MinIO, click the top option (object browser) on left, which should show central, mlflow and workers buckets. This means that the integration is succesful, which should allow you to run the demonstration notebook without problems. 

### Docker Compose Setup

Open up one terminal and move it to the following repository path:

- FFD/code/research/deployment

Run the following command to make the stack run:

```
docker compose -f ffd-docker-compose.yaml up
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

We will follow the Oakestra README instructions for setting up the orhestrator and worker nodes:

#### Orchestrator

Git clone the Oakestra repository:

```
git clone https://github.com/oakestra/oakestra.git 
cd oakestra
```

Set the following enviromental variables:

```
export CLUSTER_NAME=ffd-1
export CLUSTER_LOCATION=60.204478,24.962756,3000
export SYSTEM_MANAGER_URL=(Public IP)
```

Open a terminal and start the oakestra containers with:

```
sudo -E docker compose -f run-a-cluster/1-DOC.yaml -f run-a-cluster/override-alpha-versions.yaml up
```

#### Worker Node

In the devices of your choosing, install worker node NodeEngine

```
wget -c https://github.com/oakestra/oakestra/releases/download/alpha-v0.4.300/NodeEngine_$(dpkg --print-architecture).tar.gz && tar -xzf NodeEngine_$(dpkg --print-architecture).tar.gz && chmod +x install.sh && mv NodeEngine NodeEngine_$(dpkg --print-architecture) && ./install.sh $(dpkg --print-architecture)
```

and NetManager

```
wget -c https://github.com/oakestra/oakestra-net/releases/download/alpha-v0.4.300/NetManager_$(dpkg --print-architecture).tar.gz && tar -xzf NetManager_$(dpkg --print-architecture).tar.gz && chmod +x install.sh && ./install.sh $(dpkg --print-architecture)
```

Before you start NodeEngine and Netmanager, modify the netcfg.json using

```
sudo nano /etc/netmanager/netcfg.json
```

into the following format

```
{
  "NodePublicAddress": "(Public IP)",
  "NodePublicPort": 50103,
  "ClusterUrl": "(Public IP)",
  "ClusterMqttPort": "10003"
}
```

Now, open two terminals. Start the NetManager with

```
sudo NetManager -p 6000
```

and NodeEngine with

```
sudo NodeEngine -n 6000 -p 10100 -a (Public IP)
```

#### Checks

Now, when everything is running, go to the following addresses:

- http://(public IP)
  - Username = Admin
  - Password = Admin
- http://(Public IP):10000/api/docs

In the later case, go down the page to find Clusters and press try it out to check, if the amount of active_nodes is 1. If it is, then Oakestra is ready. 


#### FFD setup

Open up a code editor and go to:

- FFD/code/research/deployment/oakestra

