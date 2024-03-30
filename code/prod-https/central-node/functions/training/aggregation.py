import numpy as np
import torch 
import os 
from collections import OrderedDict
import time
import psutil

from functions.management.storage import store_metrics_and_resources
from functions.platforms.minio import get_object_data_and_metadata, create_or_update_object, get_object_list
from functions.training.model import FederatedLogisticRegression, test
from torch.utils.data import DataLoader
from functions.platforms.mlflow import update_run, end_run
from functions.general import format_metadata_dict

# Created and works
def get_model_updates(
    logger: any,
    minio_client: any,
    current_experiment: int,
    current_cycle: int
) -> any:
    experiments_folder = 'experiments'
    central_bucket = 'central'
    experiment_folder_path = experiments_folder + '/' + str(current_experiment)
    cycle_folder_path = experiment_folder_path + '/' + str(current_cycle)
    local_models_folder = cycle_folder_path + '/local-models'
    local_model_names = get_object_list(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        path_prefix = local_models_folder
    )
    updates = []
    collective_sample_size = 0
    for local_model_path in local_model_names.keys():
        #print(local_model_path)
        pkl_removal = local_model_path[:-4]
        #print(pkl_removal)
        local_model_object = get_object_data_and_metadata(
            logger = logger,
            minio_client = minio_client,
            bucket_name = central_bucket,
            object_path = pkl_removal
        )
        local_model_parameters = local_model_object['data']
        local_model_metadata = local_model_object['metadata']
        train_amount = int(local_model_metadata['train-amount'])
        updates.append({
            'parameters': local_model_parameters,
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
# Refactored and works
def update_global_model(
    file_lock: any,
    logger: any,
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

    if central_status['complete']:
        return False

    if not central_status['sent']:
        return False

    if central_status['updated']:
        return False
    
    os.environ['STATUS'] = 'updating global model'
    logger.info('Updating global model')

    experiment_folder_path = experiments_folder + '/' + str(central_status['experiment'])
    
    parameters_folder_path = experiment_folder_path + '/parameters'
    central_parameters_path = parameters_folder_path + '/central'
    
    central_parameters_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_parameters_path
    )
    central_parameters = central_parameters_object['data']
    
    available_updates, collective_sample_size = get_model_updates(
        logger = logger,
        minio_client = minio_client,
        current_experiment = str(central_status['experiment']),
        current_cycle = str(central_status['cycle'])
    )
    # Could be reconsidered
    if not central_parameters['min-update-amount'] <= len(available_updates):
        return False
    
    new_global_model_path = experiment_folder_path + '/' + str(central_status['cycle'] + 1) + '/global-model'
    model_parameters = model_fed_avg(
        updates = available_updates,
        total_sample_size = collective_sample_size 
    )
    model_metadata = {
        'update-amount': str(len(available_updates)),
        'train-amount': str(collective_sample_size),
        'test-amount': str(0),
        'eval-amount': str(0)
    }
    
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = new_global_model_path,
        data = model_parameters,
        metadata = model_metadata
    )
    
    central_status['collective-amount'] = collective_sample_size
    central_status['updated'] = True
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path,
        data = central_status,
        metadata = {}
    )

    os.environ['STATUS'] = 'global model updated'
    logger.info('Global model updated')

    time_end = time.time()
    cpu_end = this_process.cpu_percent(interval=0.2)
    mem_end = psutil.virtual_memory().used 
    disk_end = psutil.disk_usage('.').used
    
    time_diff = (time_end - time_start) 
    cpu_diff = cpu_end - cpu_start 
    mem_diff = (mem_end - mem_start)
    disk_diff = (disk_end - disk_start)

    resource_metrics = {
        'name': 'update-global-model',
        'time-seconds': round(time_diff,5),
        'cpu-percentage': cpu_diff,
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
        area = 'function',
        metrics = resource_metrics
    )

    return True
# Refactored
def evalute_global_model(
    file_lock: any,
    logger: any,
    minio_client: any,
    mlflow_client: any,
    prometheus_registry: any,
    prometheus_metrics: any
):
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
    
    if not central_status['updated']:
        return False
    
    if central_status['evaluated']:
        return False
    
    os.environ['STATUS'] = 'evaluating global model'
    logger.info('Evaluating global model')

    experiment_folder_path = experiments_folder + '/' + str(central_status['experiment'])
    parameters_folder_path = experiment_folder_path + '/parameters'
    central_parameters_path = parameters_folder_path + '/central'

    artifact_folder = 'artifacts'
    mlflow_parameters = {}
    mlflow_metrics = {}
    mlflow_artifacts = []

    central_parameters_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_parameters_path
    )
    central_parameters = central_parameters_object['data']
    
    model_parameters_path = parameters_folder_path + '/model' 
    model_parameters_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = model_parameters_path
    )
    model_parameters = model_parameters_object['data']

    global_model_path = experiment_folder_path + '/' + str(central_status['cycle'] + 1) + '/global-model'
    global_model_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = global_model_path
    )
    global_model_parameters = global_model_object['data']
    global_model_metadata = format_metadata_dict(global_model_object['metadata'])
    model_temp_path = artifact_folder + '/global-model' + '.pth'
    torch.save(global_model_parameters, model_temp_path)
    mlflow_artifacts.append(model_temp_path)

    model = FederatedLogisticRegression(dim = model_parameters['input-size'])
    mlflow_parameters['updates'] = int(global_model_metadata['update-amount'])
    mlflow_parameters['train-amount'] = int(central_status['collective-amount'])
    mlflow_parameters['test-amount'] = 0
    mlflow_parameters['input-size'] = int(model_parameters['input-size'])
    model.apply_parameters(model, global_model_parameters)

    eval_tensor_path = experiment_folder_path + '/tensors/eval'
    eval_tensor_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = eval_tensor_path
    )
    eval_tensor = eval_tensor_object['data']
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
        mlflow_metrics['eval-' + str(key)] = value
    
    succesful_metrics = 0
    thresholds = central_parameters['metric-thresholds']
    conditions = central_parameters['metric-conditions']
    for key,value in eval_metrics.items():
        if 'amount' in key:
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

    eval_metrics['train-amount'] = int(global_model_metadata['train-amount'])
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

    times_path = experiment_folder_path + '/times' 
    
    times_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = times_path 
    )
    times = times_object['data']

    cycle_start = times[str(central_status['cycle']-1)]['cycle-time-start']
    cycle_end = time.time()
    cycle_total = cycle_end-cycle_start
    times[str(central_status['cycle']-1)]['cycle-time-end'] = cycle_end
    times[str(central_status['cycle']-1)]['cycle-total-seconds'] = cycle_total
    if not central_status['complete']:
        times[str(central_status['cycle'])] = {
            'cycle-time-start':time.time(),
            'cycle-time-end': 0,
            'cycle-total-seconds': 0
        }

    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = times_path,
        data = times,
        metadata = {}
    )

    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path,
        data = central_status,
        metadata = {}
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
    cpu_end = this_process.cpu_percent(interval=0.2)
    mem_end = psutil.virtual_memory().used 
    disk_end = psutil.disk_usage('.').used
    
    time_diff = (time_end - time_start) 
    cpu_diff = cpu_end - cpu_start 
    mem_diff = (mem_end - mem_start)
    disk_diff = (disk_end - disk_start)

    resource_metrics = {
        'name': 'evaluate-global-model',
        'time-seconds': round(time_diff,5),
        'cpu-percentage': cpu_diff,
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
        area = 'function',
        metrics = resource_metrics
    )

    return True