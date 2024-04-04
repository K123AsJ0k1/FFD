import os 
import time

import numpy as np
import torch 

from collections import OrderedDict

from torch.utils.data import DataLoader

from functions.management.objects import get_experiments_objects, set_experiments_objects, set_object_paths

from functions.platforms.minio import get_object_list
from functions.platforms.mlflow import update_run, end_run

from functions.management.storage import store_metrics_resources_and_times
from functions.training.model import FederatedLogisticRegression, test

# Refactored and works
def get_model_updates(
    file_lock: any,
    logger: any,
    minio_client: any
) -> any:
    central_bucket = 'central'
    local_models_folder = set_object_paths()['local-models'][:-8]
    local_model_names = get_object_list(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        path_prefix = local_models_folder
    )
    updates = []
    collective_sample_size = 0
    for local_model_path in local_model_names.keys():
        pkl_removal = local_model_path[:-4]
        model_name = pkl_removal.split('/')[-1]
        
        local_model, details = get_experiments_objects(
            file_lock = file_lock,
            logger = logger,
            minio_client = minio_client,
            object = 'local-models',
            replacer = model_name
        )

        train_amount = int(details['train-amount'])
        updates.append({
            'parameters': local_model,
            'train-amount': train_amount,
        })
        collective_sample_size = collective_sample_size + train_amount
    return updates, collective_sample_size
# Refactored and works
def model_fed_avg(
    updates: any,
    total_sample_size: int    
) -> any:
    weights = []
    biases = []
    for update in updates:
        parameters = update['parameters']
        worker_sample_size = update['train-amount']
        
        worker_weights = np.array(parameters['linear.weight'].tolist()[0])
        worker_bias = np.array(parameters['linear.bias'].tolist()[0])
        
        adjusted_worker_weights = worker_weights * (worker_sample_size/total_sample_size)
        adjusted_worker_bias = worker_bias * (worker_sample_size/total_sample_size)
        
        weights.append(adjusted_worker_weights.tolist())
        biases.append(adjusted_worker_bias)
    
    FedAvg_weight = [np.sum(weights,axis = 0)]
    FedAvg_bias = [np.sum(biases, axis = 0)]

    updated_global_model = OrderedDict([
        ('linear.weight', torch.tensor(FedAvg_weight,dtype=torch.float32)),
        ('linear.bias', torch.tensor(FedAvg_bias,dtype=torch.float32))
    ])
    return updated_global_model
# Refactored
def update_global_model(
    file_lock: any,
    logger: any,
    minio_client: any,
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

    if central_status['complete']:
        return False

    if not central_status['sent']:
        return False

    if central_status['updated']:
        return False
    
    os.environ['STATUS'] = 'updating global model'
    logger.info('Updating global model')
    
    central_parameters, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'parameters',
        replacer = 'central'
    )

    available_updates, collective_sample_size = get_model_updates(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client
    )
    # Could be reconsidered
    if not central_parameters['min-update-amount'] <= len(available_updates):
        return False
    
    model_data = model_fed_avg(
        updates = available_updates,
        total_sample_size = collective_sample_size 
    )
    model_metadata = {
        'update-amount': str(len(available_updates)),
        'train-amount': str(collective_sample_size),
        'test-amount': str(0),
        'eval-amount': str(0)
    }
    set_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'updated-model',
        replacer = '',
        overwrite = True,
        object_data = model_data,
        object_metadata = model_metadata
    )
    
    central_status['collective-amount'] = collective_sample_size
    central_status['updated'] = True
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

    os.environ['STATUS'] = 'global model updated'
    logger.info('Global model updated')

    time_end = time.time()
    time_diff = (time_end - time_start) 

    resource_metrics = {
        'name': 'update-global-model',
        'action-time-start': time_start,
        'action-time-end': time_end,
        'action-total-seconds': round(time_diff,5),
    }

    store_metrics_resources_and_times(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        prometheus_registry = prometheus_registry,
        prometheus_metrics = prometheus_metrics,
        type = 'times',
        area = 'function',
        metrics = resource_metrics
    )

    return True
# Refactored and works
def evalute_global_model(
    file_lock: any,
    logger: any,
    minio_client: any,
    mlflow_client: any,
    prometheus_registry: any,
    prometheus_metrics: any
):
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
    
    if not central_status['updated']:
        return False
    
    if central_status['evaluated']:
        return False
    
    os.environ['STATUS'] = 'evaluating global model'
    logger.info('Evaluating global model')

    artifact_folder = 'artifacts'
    mlflow_parameters = {}
    mlflow_metrics = {}
    mlflow_artifacts = []

    central_parameters, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'parameters',
        replacer = 'central'
    )

    model_parameters, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'parameters',
        replacer = 'model'
    )
    
    global_model, global_model_details = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'updated-model',
        replacer = ''
    ) 

    model_temp_path = artifact_folder + '/global-model' + '.pth'
    torch.save(global_model, model_temp_path)
    mlflow_artifacts.append(model_temp_path)

    model = FederatedLogisticRegression(dim = model_parameters['input-size'])

    mlflow_parameters['experiment'] = central_status['experiment']
    mlflow_parameters['cycle'] = central_status['cycle'] 
    mlflow_parameters['updates'] = int(global_model_details['update-amount'])

    for key,value in model_parameters.items():
        mlflow_parameters[key] = value

    mlflow_parameters['train-amount'] = int(central_status['collective-amount'])
    mlflow_parameters['test-amount'] = 0
    mlflow_parameters['train-batch-size'] = 0
    mlflow_parameters['test-batch-size'] = 0
    model.apply_parameters(model, global_model)

    eval_tensor, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'tensors',
        replacer = 'eval'
    )
    mlflow_parameters['eval-amount'] = len(eval_tensor)
    eval_tensor_temp_path = artifact_folder + '/eval.pt'
    mlflow_artifacts.append(eval_tensor_temp_path)
    
    eval_batch_size = 64
    eval_loader = DataLoader(
        dataset = eval_tensor, 
        batch_size = eval_batch_size
    )
    mlflow_parameters['eval-batch-size'] = eval_batch_size

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
    
    succesful_metrics = 0
    thresholds = central_parameters['metric-thresholds']
    conditions = central_parameters['metric-conditions']
    for key,value in eval_metrics.items():
        if 'amount' in key or 'name' in key:
            continue
        message = 'Metric ' + str(key)
        if conditions[key] == '>=' and thresholds[key] <= value:
            message = message + ' succeeded with ' + str(value) + str(conditions[key]) + str(thresholds[key])
            logger.info(message)
            succesful_metrics += 1
            continue
        if conditions[key] == '<=' and value <= thresholds[key]:
            message = message + ' succeeded with ' + str(value) + str(conditions[key]) + str(thresholds[key])
            logger.info(message)
            succesful_metrics += 1
            continue
        message = message + ' failed with ' + str(value) + str(conditions[key]) + str(thresholds[key])
        logger.info(message)

    eval_metrics['train-amount'] = int(global_model_details['train-amount'])
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

    update_run(
        logger = logger,
        mlflow_client = mlflow_client,
        run_id = central_status['run-id'],
        parameters = mlflow_parameters,
        metrics = mlflow_metrics,
        artifacts = mlflow_artifacts
    )
 
    central_status['evaluated'] = True
    if central_parameters['min-metric-success'] <= succesful_metrics or central_status['cycle'] == central_parameters['max-cycles']:
        message = 'Global model achieved ' + str(succesful_metrics) + '/' + str(central_parameters['min-metric-success']) + ' of metrics in ' + str(central_status['cycle']) + '/' + str(central_parameters['max-cycles']) + ' of cycles'
        logger.info(message)
        central_status['complete'] = True
        central_status['sent'] = False
        central_status['cycle'] = central_status['cycle'] + 1
    else: 
        message = 'Global model failed ' + str(succesful_metrics) + '/' + str(central_parameters['min-metric-success']) + ' of metrics in ' + str(central_status['cycle']) + '/' + str(central_parameters['max-cycles']) + ' of cycles'
        logger.info(message)
        central_status['worker-split'] = False
        central_status['sent'] = False
        central_status['updated'] = False
        central_status['evaluated'] = False
        central_status['worker-updates'] = 0
        central_status['cycle'] = central_status['cycle'] + 1
    
    experiment_times, _ = get_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'experiment-times',
        replacer = ''
    )

    cycle_start = experiment_times[str(central_status['cycle']-1)]['cycle-time-start']
    cycle_end = time.time()
    cycle_total = cycle_end-cycle_start
    experiment_times[str(central_status['cycle']-1)]['cycle-time-end'] = cycle_end
    experiment_times[str(central_status['cycle']-1)]['cycle-total-seconds'] = cycle_total
    if not central_status['complete']:
        experiment_times[str(central_status['cycle'])] = {
            'cycle-time-start':time.time(),
            'cycle-time-end': 0,
            'cycle-total-seconds': 0
        }

    set_experiments_objects(
        file_lock = file_lock,
        logger = logger,
        minio_client = minio_client,
        object = 'experiment-times',
        replacer = '',
        overwrite = True,
        object_data = experiment_times,
        object_metadata = {}
    )

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
    
    os.environ['STATUS'] = 'global model evaluated'
    logger.info('Global model evaluated')

    end_run(
        logger = logger,
        mlflow_client = mlflow_client,
        run_id = central_status['run-id'],
        status = 'FINISHED'
    )
    
    time_end = time.time()
    time_diff = (time_end - time_start) 
    resource_metrics = {
        'name': 'evaluate-global-model',
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
        metrics = resource_metrics
    )
    os.environ['CYCLE'] = str(central_status['cycle'])
    return True