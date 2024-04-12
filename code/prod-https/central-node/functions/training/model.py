import time
import os

import numpy as np

import torch
import torch.nn as nn 

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from functions.platforms.minio import check_object, get_object_data_and_metadata
from functions.platforms.mlflow import start_run, update_run, end_run

from functions.general import format_metadata_dict

from functions.management.objects import get_experiments_objects, set_experiments_objects, set_object_paths

from functions.management.storage import store_metrics_resources_and_times

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
    time_diff = (time_end - time_start) 

    action_time = {
        'name': 'logistic-regression-training',
        'epochs': model_parameters['epochs'],
        'batches': len(train_loader),
        'average-batch-size': total_size / len(train_loader),
        'action-time-start': time_start,
        'action-time-end': time_end,
        'action-total-seconds': round(time_diff,5)
    }

    store_metrics_resources_and_times(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        type = 'times',
        area = 'training',
        metrics = action_time
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
        time_diff = (time_end - time_start) 
        action_time = {
            'name': 'logistic-regression-' + name,
            'epochs': 0,
            'batches': len(test_loader),
            'average-batch-size': total_size / len(test_loader),
            'action-time-start': time_start,
            'action-time-end': time_end,
            'action-total-seconds': round(time_diff,5)
        }

        store_metrics_resources_and_times(
            file_lock = file_lock,
            logger = logger,
            minio_client = minio_client,
            prometheus_registry = prometheus_registry,
            prometheus_metrics = prometheus_metrics,
            type = 'times',
            area = 'training',
            metrics = action_time
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
            logger.warning(e)
        
        metrics = {
            'name': 'logistic-regression-' + name,
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
    minio_client: any,
    mlflow_client: any,
    prometheus_registry: any,
    prometheus_metrics: any
) -> bool:
    time_start = time.time()

    central_status, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = ''
    )

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
    run_name = 'initial-training-' + str(central_status['experiment']) + '-' + str(central_status['cycle'])
    run_data = start_run(
        logger = logger,
        mlflow_client = mlflow_client,
        experiment_id = central_status['experiment-id'],
        tags = {},
        name = run_name
    )
    # This is for using the with instead of client
    #mlflow.set_tracking_uri('http://127.0.0.1:5000')
    #mlflow.set_experiment(experiment_id = central_status['experiment-id'])

    model_parameters, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'parameters',
        replacer = 'model'
    )
    
    torch.manual_seed(model_parameters['seed'])
    model = FederatedLogisticRegression(dim = model_parameters['input-size'])
    mlflow_parameters['experiment'] = central_status['experiment']
    mlflow_parameters['cycle'] = central_status['cycle']
    mlflow_parameters['updates'] = 0

    for key,value in model_parameters.items():
        mlflow_parameters[key] = value

    train_tensor, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'tensors',
        replacer = 'train'
    )
    mlflow_parameters['train-amount'] = len(train_tensor)
    train_tensor_temp_path = artifact_folder + '/train.pt'
    torch.save(train_tensor, train_tensor_temp_path)
    mlflow_artifacts.append(train_tensor_temp_path)

    test_tensor, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'tensors',
        replacer = 'eval'
    )
    mlflow_parameters['test-amount'] = len(test_tensor)
    test_tensor_temp_path = artifact_folder + '/test.pt'
    torch.save(test_tensor, test_tensor_temp_path)
    mlflow_artifacts.append(test_tensor_temp_path)

    eval_tensor, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'tensors',
        replacer = 'eval'
    )
    mlflow_parameters['eval-amount'] = len(eval_tensor)
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

    for key,value in test_metrics.items():
        if not key == 'name':
            mlflow_metrics['test-' + str(key)] = value

    test_metrics['train-amount'] = len(train_tensor)
    test_metrics['test-amount'] = len(test_tensor)
    test_metrics['eval-amount'] = 0

    store_metrics_resources_and_times(
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

    for key,value in eval_metrics.items():
        if not key == 'name':
            mlflow_metrics['eval-' + str(key)] = value

    eval_metrics['train-amount'] = len(train_tensor)
    eval_metrics['test-amount'] = 0
    eval_metrics['eval-amount'] = len(eval_tensor)
    store_metrics_resources_and_times(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        type = 'metrics',
        area = '',
        metrics = eval_metrics
    )

    model_data = model.get_parameters(model)
    model_metadata = {
        'updates': 0,
        'train-amount': str(len(train_tensor)),
        'test-amount':  str(len(test_tensor)),
        'eval-amount':  str(len(eval_tensor)),
    }
    set_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'global-model',
        replacer = '',
        overwrite = True,
        object_data = model_data,
        object_metadata = model_metadata
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
    set_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'status',
        replacer = '',
        overwrite = True,
        object_data = central_status,
        object_metadata = {}
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
    time_diff = (time_end - time_start) 
    
    action_time = {
        'name': 'initial-model-training',
        'action-time-start': time_start,
        'action-time-end': time_end,
        'action-total-seconds': round(time_diff,5)
    }

    store_metrics_resources_and_times(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        type = 'times',
        area = 'function',
        metrics = action_time
    )
    return True
# Refactored and works
def model_inference(
    file_lock: any,
    logger: any,
    minio_client: any,
    prometheus_registry: any,
    prometheus_metrics: any,
    experiment_name: str,
    experiment: str,
    cycle: str,
    input: any
) -> any:
    used_bucket = 'central'
    time_start = time.time()
    
    global_model_path = set_object_paths()['global-model']
    
    path_split = global_model_path.split('/')
    path_split[1] = experiment_name
    path_split[2] = experiment
    path_split[3] = cycle
    wanted_global_model_path = '/'.join(path_split)
    
    model_data = None
    model_metadata = None
    with file_lock:
        object_exists = check_object(
            logger = logger,
            minio_client = minio_client,
            bucket_name = used_bucket,
            object_path = wanted_global_model_path
        )
        
        if object_exists:
            fetched_object = get_object_data_and_metadata(
                logger = logger,
                minio_client = minio_client,
                bucket_name = used_bucket,
                object_path = wanted_global_model_path
            )
            
            model_data = fetched_object['data']
            model_metadata = format_metadata_dict(fetched_object['metadata'])

    output = None
    if not model_data is None:
        model_parameters, _ = get_experiments_objects(
            file_lock = file_lock,
            logger = logger,
            minio_client = minio_client,
            object = 'parameters',
            replacer = 'model'
        )

        model = FederatedLogisticRegression(dim = model_parameters['input-size'])
        model.apply_parameters(model, model_data)

        input_tensor = torch.tensor(np.array(input, dtype=np.float32))
        prediction = None
        with torch.no_grad():
            prediction = model.prediction(model,input_tensor)
        
        output = prediction.tolist()
        
    time_end = time.time()
    time_diff = (time_end - time_start) 
    
    action_time = {
        'name': experiment_name + '-' + experiment + '-' + cycle + '-prediction',
        'sample-amount': len(input_tensor),
        'action-time-start': time_start,
        'action-time-end': time_end,
        'action-total-seconds': round(time_diff,5)
    }

    store_metrics_resources_and_times(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        type = 'times',
        area = 'inference',
        metrics = action_time
    )

    return output
