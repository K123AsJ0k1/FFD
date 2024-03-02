from flask import current_app
import torch
import torch.nn as nn 
import psutil

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from functions.data_functions import * 
from functions.storage_functions import *
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
def get_train_test_loaders(
    train_tensor: any,
    test_tensor: any,
    model_parameters: any 
) -> any:
    train_loader = DataLoader(
        train_tensor,
        batch_size = int(len(train_tensor) * model_parameters['sample-rate']),
        generator = torch.Generator().manual_seed(model_parameters['seed'])
    )
    test_loader = DataLoader(test_tensor, 64)
    return train_loader,test_loader
# Refactored and works
def train(
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
    
    for epoch in range(model_parameters['epochs']):
        losses = []
        for batch in train_loader:
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

    time_diff = (time_end - time_start) # seconds
    cpu_diff = cpu_end - cpu_start # percentage
    mem_diff = (mem_end - mem_start) / (1024 ** 2) # megabytes
    disk_diff = (disk_end - disk_start) / (1024 ** 2) # megabytes

    average_time = time_diff / model_parameters['epochs']
    average_cpu = cpu_diff / model_parameters['epochs']
    average_mem = mem_diff / model_parameters['epochs']
    average_diffs = disk_diff / model_parameters['epochs']

    resource_metrics = {
        'model': 'logistic-regression',
        'epochs': model_parameters['epochs'],
        'average-epoch-time-seconds': round(average_time,5),
        'average-epoch-cpu-percentage': round(average_cpu,5),
        'average-epoch-ram-megabytes': round(average_mem,5),
        'average-epoch-disk-megabytes': round(average_diffs,5)
    }

    status = store_metrics(
        type = 'resources',
        subject = 'training',
        metrics = resource_metrics
    )
    
# Refactored and works
def test(
    model: any, 
    test_loader: any
) -> any:
    with torch.no_grad():
        total_size = 0
        total_confusion_matrix = [0,0,0,0]
        
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
    logger: any
) -> bool:
    current_experiment_number = get_current_experiment_number()
    central_status_path = 'status/experiment_' + str(current_experiment_number) + '/central.txt'
    if not os.path.exists(central_status_path):
        return False
    
    central_status = None
    with open(central_status_path, 'r') as f:
        central_status = json.load(f)

    if not central_status['start']:
        return False

    if not central_status['data-split'] or not central_status['preprocessed']:
        return False

    if central_status['trained']:
        return False
    
    this_process = psutil.Process(os.getpid())
    
    mem_start = psutil.virtual_memory().used 
    disk_start = psutil.disk_usage('.').used
    cpu_start = this_process.cpu_percent(interval=0.2)
    time_start = time.time()
    
    model_parameters_path = 'parameters/experiment_' + str(current_experiment_number) + '/model.txt'
    model_parameters = None
    with open(model_parameters_path, 'r') as f:
        model_parameters = json.load(f)

    tensor_folder_path = 'tensors/experiment_' + str(current_experiment_number)
    train_tensor_path = tensor_folder_path +  '/train_0.pt'
    test_tensor_path = tensor_folder_path +  '/test_0.pt'

    train_tensor = torch.load(train_tensor_path)
    test_tensor = torch.load(test_tensor_path)

    torch.manual_seed(model_parameters['seed'])
    given_train_loader, given_test_loader = get_train_test_loaders(
        train_tensor = train_tensor,
        test_tensor = test_tensor,
        model_parameters = model_parameters
    )
    lr_model = FederatedLogisticRegression(dim = model_parameters['input-size'])
    
    train(
        logger = logger,
        model = lr_model, 
        train_loader = given_train_loader,  
        model_parameters = model_parameters
    )
    
    test_metrics = test(
        model = lr_model, 
        test_loader = given_test_loader
    )
    
    test_metrics['train-amount'] = len(train_tensor)
    test_metrics['test-amount'] = len(test_tensor)
    status = store_metrics(
        type = 'global',
        subject = '',
        metrics = test_metrics
    )
    
    #The used format for models is global_(cycle)_(updates)_(samples).pth
    model_experiment_folder = 'models/experiment_' + str(current_experiment_number)
    os.makedirs(model_experiment_folder, exist_ok = True)
    model_path = model_experiment_folder + '/global_0_0_' + str(central_status['train-amount']) + '.pth'
    parameters = lr_model.get_parameters(lr_model)
    torch.save(parameters, model_path)
    
    central_status['trained'] = True
    with open(central_status_path, 'w') as f:
        json.dump(central_status, f, indent=4) 

    time_end = time.time()
    cpu_end = this_process.cpu_percent(interval=0.2)
    mem_end = psutil.virtual_memory().used 
    disk_end = psutil.disk_usage('.').used

    time_diff = (time_end - time_start) # seconds
    cpu_diff = cpu_end - cpu_start # percentage
    mem_diff = (mem_end - mem_start) / (1024 ** 2) # megabytes
    disk_diff = (disk_end - disk_start) / (1024 ** 2) # megabytes

    resource_metrics = {
        'name': 'initial-model-training',
        'time-seconds': time_diff,
        'cpu-percentage': cpu_diff,
        'ram-megabytes': mem_diff,
        'disk-megabytes': disk_diff
    }

    status = store_metrics(
        type = 'resources',
        subject = 'function',
        metrics = resource_metrics
    )

    return True
# Refactor
def model_inference(
    input: any,
    cycle: int
) -> any:
    training_status_path = 'logs/training_status.txt'
    if not os.path.exists(training_status_path):
        return False
    # Might be useful for recoding inference amounts
    training_status = None
    with open(training_status_path, 'r') as f:
        training_status = json.load(f)

    GLOBAL_PARAMETERS = current_app.config['GLOBAL_PARAMETERS']

    files = os.listdir('models')

    wanted_global_model = ''
    for file in files:
        first_split = file.split('.')
        second_split = first_split[0].split('_')
        name = str(second_split[0])
        if name == 'global':
            model_cycle = int(second_split[1])
            if cycle == model_cycle:
                wanted_global_model = 'models/' + file 

    if not os.path.exists(wanted_global_model):
        return None
    
    given_parameters = torch.load(wanted_global_model)
    
    lr_model = FederatedLogisticRegression(dim = GLOBAL_PARAMETERS['input-size'])
    lr_model.apply_parameters(lr_model, given_parameters)
    
    given_input = torch.tensor(np.array(input, dtype=np.float32))

    with torch.no_grad():
        output = lr_model.prediction(lr_model,given_input)

    return output.tolist()
# Refactor
def get_models() -> any:
    training_status_path = 'logs/training_status.txt'
    if not os.path.exists(training_status_path):
        return False
    
    training_status = None
    with open(training_status_path, 'r') as f:
        training_status = json.load(f)

    models_folder_path = 'models'
    files = os.listdir(models_folder_path)
    if len(files) == 0:
        return None
    stored_models = {
        'global': {},
        'workers': {}
    }
    for model_key in range(0, training_status['parameters']['cycle']):
        stored_models['global'] = {str(model_key): {}}
    for worker_key in training_status['workers'].keys():
        stored_models['workers'] = {str(worker_key): {}}
        for metric_key in range(0, training_status['parameters']['cycle']):
            stored_models['workers'][str(worker_key)] = {str(metric_key): {}}

    for file in files:
        first_split = file.split('.')[0]
        second_split = first_split.split('_')
        model = torch.load(models_folder_path + '/' + file) 
        formatted_local_model = {
            'weights': model['linear.weight'].numpy().tolist(),
            'bias': model['linear.bias'].numpy().tolist()
        }
        if second_split[0] == 'global':
            cycle = str(second_split[1])
            updates = str(second_split[2])
            train_amount = str(second_split[3])
            stored_models['global'][cycle] = {
                'parameters': formatted_local_model,
                'updates': updates,
                'train-amount': train_amount
            }
        if second_split[0] == 'worker':
            id = str(second_split[1])
            cycle = str(second_split[2])
            train_amount = str(second_split[3])
            stored_models['workers'][id][cycle] = {
                'parameters': formatted_local_model,
                'train-amount': train_amount
            }
    return stored_models