from flask import current_app

import torch
import torch.nn as nn 
import psutil
import time
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from functions.platforms.minio import get_object_data_and_metadata, create_or_update_object
from functions.general import format_metadata_dict
from functions.management.storage import store_metrics_and_resources
from functions.platforms.mlflow import start_run, update_run, end_run

import os
import mlflow

# Refactored and works
class FederatedLogisticRegression(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1, bias=bias)
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, x):
        return self.linear(x).view(-1)

    @staticmethod
    def train_step(model, batch):
        x, y = batch
        out = model(x)
        loss = model.loss(out, y)
        return loss

    @staticmethod
    def test_step(model, batch):
        x, y = batch
        out = model(x)
        loss = model.loss(out, y)
        preds = out > 0 # Predict y = 1 if P(y = 1) > 0.5
        return loss, preds
    
    @staticmethod
    def prediction(model, x):
        out = model(x)
        preds = out > 0
        return preds

    @staticmethod
    def get_parameters(model):
        return model.state_dict()

    @staticmethod
    def apply_parameters(model, parameters):
        model.load_state_dict(parameters)
# Refactored and works
def train(
    file_lock: any,
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any,
    model: any,
    train_loader: any,
    model_parameters: any
):
    opt_func = None
    if model_parameters['optimizer'] == 'SGD':
        opt_func = torch.optim.SGD

    optimizer = opt_func(model.parameters(), model_parameters['learning-rate'])
    model_type = type(model)
    
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()
    total_size = 0
    for epoch in range(model_parameters['epochs']):
        losses = []
        for batch in train_loader:
            total_size += len(batch[1])
            loss = model_type.train_step(model, batch)
            loss.backward()
            losses.append(loss)
            optimizer.step()
            optimizer.zero_grad()
        loss = torch.sum(loss) / len(train_loader)
        loss_value = loss.item()
        logger.info('Epoch ' + str(epoch + 1) + ', loss = ' + str(loss_value))

    time_end = time.time()
    cpu_end = this_process.cpu_percent(interval=0.2)
    mem_end = psutil.virtual_memory().used 
    disk_end = psutil.disk_usage('.').used

    time_diff = (time_end - time_start) 
    cpu_diff = cpu_end - cpu_start 
    mem_diff = (mem_end - mem_start) 
    disk_diff = (disk_end - disk_start) 

    resource_metrics = {
        'name': 'logistic-regression-training',
        'epochs': model_parameters['epochs'],
        'batches': len(train_loader),
        'average-batch-size': total_size / len(train_loader),
        'time-seconds': round(time_diff,5),
        'cpu-percentage': round(cpu_diff,5),
        'ram-bytes': round(mem_diff,5),
        'disk-bytes': round(disk_diff,5)
    }

    store_metrics_and_resources(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        type = 'resources',
        area = 'training',
        metrics = resource_metrics
    )
# Refactored and works
def test(
    file_lock: any,
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any,
    model: any, 
    test_loader: any,
    name: any
) -> any:
    with torch.no_grad():
        total_confusion_matrix = [0,0,0,0]
        
        this_process = psutil.Process(os.getpid())
        mem_start = psutil.virtual_memory().used 
        disk_start = psutil.disk_usage('.').used
        cpu_start = this_process.cpu_percent(interval=0.2)
        time_start = time.time()
        
        total_size = 0
        for batch in test_loader:
            total_size += len(batch[1])
            _, correct = batch
            _, preds = model.test_step(model, batch)
            
            formated_correct = correct.numpy()
            formated_preds = preds.numpy().astype(int)
            
            tn, fp, fn, tp = confusion_matrix(
                y_true = formated_correct,
                y_pred = formated_preds,
                labels = [0,1]
            ).ravel()

            total_confusion_matrix[0] += int(tp) # True positive
            total_confusion_matrix[1] += int(fp) # False positive
            total_confusion_matrix[2] += int(tn) # True negative
            total_confusion_matrix[3] += int(fn) # False negative
            
        time_end = time.time()
        cpu_end = this_process.cpu_percent(interval=0.2)
        mem_end = psutil.virtual_memory().used 
        disk_end = psutil.disk_usage('.').used

        time_diff = (time_end - time_start) 
        cpu_diff = cpu_end - cpu_start 
        mem_diff = (mem_end - mem_start)
        disk_diff = (disk_end - disk_start)

        resource_metrics = {
            'name': 'logistic-regression-' + name,
            'batches': len(test_loader),
            'average-batch-size': total_size / len(test_loader),
            'time-seconds': round(time_diff,5),
            'cpu-percentage': round(cpu_diff,5),
            'ram-bytes': round(mem_diff,5),
            'disk-bytes': round(disk_diff,5)
        }

        store_metrics_and_resources(
            file_lock = file_lock,
            logger = logger,
            minio_client = minio_client,
            prometheus_registry = prometheus_registry,
            prometheus_metrics = prometheus_metrics,
            type = 'resources',
            area = 'training',
            metrics = resource_metrics
        )
    
        TP, FP, TN, FN = total_confusion_matrix
        # Zero divsion can happen
        TPR = 0
        TNR = 0
        PPV = 0
        FNR = 0
        FPR = 0
        BA = 0
        ACC = 0
        try:
            TPR = TP/(TP+FN)
            TNR = TN/(TN+FP)
            PPV = TP/(TP+FP)
            FNR = FN/(FN+TP)
            FPR = FP/(FP+TN)
            BA = (TPR+TNR)/2
            ACC = (TP + TN)/(TP + TN + FP + FN)
        except Exception as e:
            current_app.logger.warning(e)
        
        metrics = {
            'true-positives': TP,
            'false-positives': FP,
            'true-negatives': TN,
            'false-negatives': FN,
            'recall': float(round(TPR,5)),
            'selectivity': float(round(TNR,5)),
            'precision': float(round(PPV,5)),
            'miss-rate': float(round(FNR,5)),
            'fall-out': float(round(FPR,5)),
            'balanced-accuracy': float(round(BA,5)),
            'accuracy': float(round(ACC,5))
        }
        
        return metrics
# Refactored and works
def initial_model_training(
    file_lock: any,
    logger: any,
    mlflow_client: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any
) -> bool:
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    experiments_folder = 'experiments'
    central_bucket = 'central'
    central_status_path = experiments_folder + '/status'
    central_status_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path
    )
    central_status = central_status_object['data']

    if central_status is None:
        return False

    if not central_status['start']:
        return False

    if not central_status['data-split'] or not central_status['preprocessed']:
        return False

    if central_status['trained']:
        return False

    os.environ['STATUS'] = 'training initial model'
    logger.info('Training initial model')

    artifact_folder = 'artifacts'
    mlflow_parameters = {}
    mlflow_metrics = {}
    mlflow_artifacts = []
    
    run_data = start_run(
        logger = logger,
        mlflow_client = mlflow_client,
        experiment_id = central_status['experiment-id'],
        tags = {},
        name = 'central-initial-training'
    )
    # Is needed to save artifacts
    #os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
    #os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
    #os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
    # This is for using the with instead of client
    #mlflow.set_tracking_uri('http://127.0.0.1:5000')
    #mlflow.set_experiment(experiment_id = central_status['experiment-id'])

    experiment_folder = experiments_folder + '/' + str(central_status['experiment'])
    parameters_folder_path = experiment_folder + '/parameters'
    model_parameters_path = parameters_folder_path + '/model' 
    model_parameters_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = model_parameters_path
    )
    model_parameters = model_parameters_object['data']
    
    torch.manual_seed(model_parameters['seed'])
    model = FederatedLogisticRegression(dim = model_parameters['input-size'])
    mlflow_parameters['seed'] = model_parameters['seed']
    mlflow_parameters['input-size'] = model_parameters['input-size']
    
    tensors_folder_path = experiment_folder + '/tensors'
    train_tensor_path = tensors_folder_path  + '/train'
    train_tensor_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = train_tensor_path
    )
    train_tensor = train_tensor_object['data']
    train_tensor_temp_path = artifact_folder + '/train.pt'
    torch.save(train_tensor, train_tensor_temp_path)
    mlflow_artifacts.append(train_tensor_temp_path)

    test_tensor_path = tensors_folder_path  + '/test'
    test_tensor_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = test_tensor_path
    )
    test_tensor = test_tensor_object['data']
    test_tensor_temp_path = artifact_folder + '/test.pt'
    torch.save(test_tensor, test_tensor_temp_path)
    mlflow_artifacts.append(test_tensor_temp_path)

    eval_tensor_path = tensors_folder_path  + '/eval'
    eval_tensor_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = eval_tensor_path
    )
    eval_tensor = eval_tensor_object['data']
    eval_tensor_temp_path = artifact_folder + '/eval.pt'
    torch.save(test_tensor, eval_tensor_temp_path)
    mlflow_artifacts.append(eval_tensor_temp_path)
    
    generator = torch.Generator().manual_seed(model_parameters['seed'])
    train_batch_size = int(len(train_tensor) * model_parameters['sample-rate'])
    test_batch_size = 64
    eval_batch_size = 64

    mlflow_parameters['train-batch-size'] = train_batch_size
    mlflow_parameters['test-batch-size'] = test_batch_size
    mlflow_parameters['eval-batch-size'] = eval_batch_size
    
    train_loader = DataLoader(
        dataset = train_tensor,
        batch_size = train_batch_size,
        generator = generator
    )
    test_loader = DataLoader(
        dataset = test_tensor, 
        batch_size = test_batch_size
    )
    eval_loader = DataLoader(
        dataset = eval_tensor, 
        batch_size= eval_batch_size
    )

    train(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        model = model,
        train_loader = train_loader,
        model_parameters = model_parameters
    )

    test_metrics = test(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        model = model,
        test_loader = test_loader,
        name = 'testing'
    )

    test_metrics['train-amount'] = len(train_tensor)
    test_metrics['test-amount'] = len(test_tensor)
    test_metrics['eval-amount'] = 0

    store_metrics_and_resources(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        type = 'metrics',
        area = '',
        metrics = test_metrics
    )

    eval_metrics = test(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        model = model,
        test_loader = eval_loader,
        name = 'evaluation'
    )

    eval_metrics['train-amount'] = len(train_tensor)
    eval_metrics['test-amount'] = 0
    eval_metrics['eval-amount'] = len(eval_tensor)
    store_metrics_and_resources(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        type = 'metrics',
        area = '',
        metrics = eval_metrics
    )

    for key,value in test_metrics.items():
        mlflow_metrics['test-' + str(key)] = value

    for key,value in eval_metrics.items():
        mlflow_metrics['eval-' + str(key)] = value

    cycle_folder_path = experiments_folder + '/' + str(central_status['experiment']) + '/' + str(central_status['cycle'])
    global_model_path = cycle_folder_path + '/global-model' 
    model_data = model.get_parameters(model)
    model_metadata = {
        'train-amount': str(len(train_tensor)),
        'test-amount':  str(len(test_tensor)),
        'eval-amount':  str(len(eval_tensor)),
    }
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = global_model_path,
        data = model_data,
        metadata = model_metadata
    )
    model_temp_path = artifact_folder + '/global-model' + '.pth'
    torch.save(model_data, model_temp_path)
    mlflow_artifacts.append(model_temp_path)

    update_run(
        logger = logger,
        mlflow_client = mlflow_client,
        run_id = run_data['id'],
        parameters = mlflow_parameters,
        metrics = mlflow_metrics,
        artifacts = mlflow_artifacts
    )
    
    central_status['trained'] = True
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path,
        data = central_status,
        metadata = {}
    )

    os.environ['STATUS'] = 'initial model trained'
    logger.info('Initial model trained')
    
    end_run(
        logger = logger,
        mlflow_client = mlflow_client,
        run_id = run_data['id'],
        status = 'FINISHED'
    )
        
    time_end = time.time()
    cpu_end = this_process.cpu_percent(interval=0.2)
    mem_end = psutil.virtual_memory().used 
    disk_end = psutil.disk_usage('.').used

    time_diff = (time_end - time_start) 
    cpu_diff = cpu_end - cpu_start 
    mem_diff = (mem_end - mem_start)
    disk_diff = (disk_end - disk_start) 

    resource_metrics = {
        'name': 'initial-model-training',
        'time-seconds': time_diff,
        'cpu-percentage': cpu_diff,
        'ram-bytes': mem_diff,
        'disk-bytes': disk_diff
    }

    store_metrics_and_resources(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        type = 'resources',
        area = 'function',
        metrics = resource_metrics
    )
    return True
# Refactor
def model_inference(
    file_lock: any,
    experiment: int,
    subject: str,
    cycle: int,
    input: any
) -> any:
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    current_experiment_number = get_current_experiment_number()
    model_parameters_path = 'parameters/experiment_' + str(current_experiment_number) + '/model.txt'
    model_parameters = get_file_data(
        file_lock = file_lock,
        file_path = model_parameters_path
    )

    wanted_model = get_wanted_model(
        file_lock = file_lock,
        experiment = experiment,
        subject = subject,
        cycle = cycle
    )
    
    lr_model = FederatedLogisticRegression(dim = model_parameters['input-size'])
    lr_model.apply_parameters(lr_model, wanted_model)
    
    input_tensor = torch.tensor(np.array(input, dtype=np.float32))

    with torch.no_grad():
        output = lr_model.prediction(lr_model,input_tensor)

    time_end = time.time()
    cpu_end = this_process.cpu_percent(interval=0.2)
    mem_end = psutil.virtual_memory().used 
    disk_end = psutil.disk_usage('.').used

    time_diff = (time_end - time_start) 
    cpu_diff = cpu_end - cpu_start 
    mem_diff = (mem_end - mem_start)
    disk_diff = (disk_end - disk_start)

    resource_metrics = {
        'name': str(experiment) + '-' + subject + '-' + str(cycle) + '-prediction',
        'sample-amount': len(input_tensor),
        'time-seconds': time_diff,
        'cpu-percentage': cpu_diff,
        'ram-bytes': mem_diff,
        'disk-bytes': disk_diff
    }

    status = store_metrics_and_resources(
        file_lock = file_lock,
        type = 'resources',
        subject = 'central',
        area = 'inference',
        metrics = resource_metrics
    )

    return output.tolist()
