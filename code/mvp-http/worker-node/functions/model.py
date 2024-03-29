from flask import current_app
import torch
import torch.nn as nn 
import psutil
import time
import os
import numpy as np

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from functions.general import get_current_experiment_number, get_wanted_model, get_file_data
from functions.storage import store_metrics_and_resources, store_file_data
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
def get_train_test_and_eval_loaders(
    train_tensor: any,
    test_tensor: any,
    eval_tensor: any,
    model_parameters: any 
) -> any:
    train_loader = DataLoader(
        train_tensor,
        batch_size = int(len(train_tensor) * model_parameters['sample-rate']),
        generator = torch.Generator().manual_seed(model_parameters['seed'])
    )
    test_loader = DataLoader(test_tensor, 64)
    eval_loader = DataLoader(eval_tensor, 64)
    return train_loader,test_loader,eval_loader
# Refactored and works
def train(
    file_lock: any,
    logger: any,
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

    status = store_metrics_and_resources(
        file_lock = file_lock,
        type = 'resources',
        subject = 'worker',
        area = 'training',
        metrics = resource_metrics
    )
# Refactored and works
def test(
    file_lock: any,
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

        status = store_metrics_and_resources(
            file_lock = file_lock,
            type = 'resources',
            subject = 'worker',
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
def local_model_training(
    file_lock: any,
    logger: any
) -> any:
    this_process = psutil.Process(os.getpid())
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()

    current_experiment_number = get_current_experiment_number()
    worker_status_path = 'status/experiment_' + str(current_experiment_number) + '/worker.txt'
    
    worker_status = get_file_data(
        file_lock = file_lock,
        file_path = worker_status_path
    )

    if worker_status is None:
        return False

    if worker_status['complete']:
        return False

    if not worker_status['stored'] or not worker_status['preprocessed']:
        return False

    if worker_status['trained']:
        return False

    os.environ['STATUS'] = 'training local model'
    logger.info('Training local model')
    
    model_parameters_path = 'parameters/experiment_' + str(current_experiment_number) + '/model.txt'

    model_parameters = get_file_data(
        file_lock = file_lock,
        file_path = model_parameters_path
    )

    if model_parameters is None:
        return False

    torch.manual_seed(model_parameters['seed'])
    
    tensor_folder_path = 'tensors/experiment_' + str(current_experiment_number)
    train_tensor_path = tensor_folder_path + '/train_' + str(worker_status['cycle']) + '.pt'
    train_tensor = get_file_data(
        file_lock = file_lock,
        file_path = train_tensor_path
    )
    test_tensor_path = tensor_folder_path + '/test_' + str(worker_status['cycle']) + '.pt'
    test_tensor = get_file_data(
        file_lock = file_lock,
        file_path = test_tensor_path
    )
    eval_tensor_path = tensor_folder_path + '/eval_' + str(worker_status['cycle']) + '.pt'
    eval_tensor = get_file_data(
        file_lock = file_lock,
        file_path = eval_tensor_path
    )

    models_folder_path = 'models/experiment_' + str(current_experiment_number)
    
    global_model_parameters = get_wanted_model(
        file_lock = file_lock,
        experiment = current_experiment_number,
        subject = 'global',
        cycle = worker_status['cycle']-1
    )

    given_train_loader, given_test_loader, given_eval_loader = get_train_test_and_eval_loaders(
        train_tensor = train_tensor,
        test_tensor = test_tensor,
        eval_tensor = eval_tensor,
        model_parameters = model_parameters
    )

    lr_model = FederatedLogisticRegression(dim = model_parameters['input-size'])
    lr_model.apply_parameters(lr_model, global_model_parameters)

    train(
        file_lock = file_lock,
        logger = logger,
        model = lr_model, 
        train_loader = given_train_loader,
        model_parameters = model_parameters
    )
    
    test_metrics = test( 
        file_lock = file_lock,
        model = lr_model, 
        test_loader = given_test_loader,
        name = 'testing'
    )
    test_metrics['train-amount'] = len(train_tensor)
    test_metrics['test-amount'] = len(test_tensor)
    test_metrics['eval-amount'] = 0
    status = store_metrics_and_resources(
        file_lock = file_lock,
        type = 'metrics',
        subject = 'local',
        area = '',
        metrics = test_metrics
    )

    test_metrics = test(
        file_lock = file_lock,
        model = lr_model, 
        test_loader = given_eval_loader,
        name = 'evalution'
    )
    
    test_metrics['train-amount'] = len(train_tensor)
    test_metrics['test-amount'] = 0
    test_metrics['eval-amount'] = len(eval_tensor)
    status = store_metrics_and_resources(
        file_lock = file_lock,
        type = 'metrics',
        subject = 'local',
        area = '',
        metrics = test_metrics
    )
    local_model_path = models_folder_path + '/local_' + str(worker_status['cycle']) + '.pth'
    local_model_parameters = lr_model.get_parameters(lr_model)
    
    store_file_data(
        file_lock = file_lock,
        replace = False,
        file_folder_path = models_folder_path,
        file_path = local_model_path,
        data = local_model_parameters
    )
    
    worker_status['trained'] = True
    store_file_data(
        file_lock = file_lock,
        replace = True,
        file_folder_path = '',
        file_path = worker_status_path,
        data = worker_status
    )
    
    os.environ['STATUS'] = 'local model trained'
    logger.info('Local model trained')

    time_end = time.time()
    cpu_end = this_process.cpu_percent(interval=0.2)
    mem_end = psutil.virtual_memory().used 
    disk_end = psutil.disk_usage('.').used

    time_diff = (time_end - time_start) 
    cpu_diff = cpu_end - cpu_start 
    mem_diff = (mem_end - mem_start) 
    disk_diff = (disk_end - disk_start)

    resource_metrics = {
        'name': 'local-model-training',
        'time-seconds': time_diff,
        'cpu-percentage': cpu_diff,
        'ram-bytes': mem_diff,
        'disk-bytes': disk_diff
    }

    status = store_metrics_and_resources(
        file_lock = file_lock,
        type = 'resources',
        subject = 'worker',
        area = 'function',
        metrics = resource_metrics
    )

    return True
# Refactored and works
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
        subject = 'worker',
        area = 'inference',
        metrics = resource_metrics
    )

    return output.tolist()