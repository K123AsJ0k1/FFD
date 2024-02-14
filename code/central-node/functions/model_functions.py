from flask import current_app
import torch
import torch.nn as nn
from collections import OrderedDict

from torch.optim import SGD

from functions.data_functions import *

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
        corrects = torch.tensor(torch.sum(preds == y).item())
        return loss, corrects

    @staticmethod
    def get_parameters(model):
        return model.state_dict()

    @staticmethod
    def apply_parameters(model, parameters):
        model.load_state_dict(parameters)

def get_train_test_loaders() -> any:
    GLOBAL_PARAMETERS = current_app.config['GLOBAL_PARAMETERS']
    
    train_tensor = torch.load('tensors/train.pt')
    test_tensor = torch.load('tensors/test.pt')

    train_loader = DataLoader(
        train_tensor,
        batch_size = int(len(train_tensor) * GLOBAL_PARAMETERS['sample-rate']),
        generator = torch.Generator().manual_seed(GLOBAL_PARAMETERS['seed'])
    )
    test_loader = DataLoader(test_tensor, 64)
    return train_loader,test_loader

def train(
    model: any,
    train_loader: any
):
    GLOBAL_PARAMETERS = current_app.config['GLOBAL_PARAMETERS']
    
    opt_func = None
    if GLOBAL_PARAMETERS['optimizer'] == 'SGD':
        opt_func = torch.optim.SGD

    optimizer = opt_func(model.parameters(), GLOBAL_PARAMETERS['learning-rate'])
    model_type = type(model)
    
    for epoch in range(GLOBAL_PARAMETERS['epochs']):
        losses = []
        for batch in train_loader:
            loss = model_type.train_step(model, batch)
            loss.backward()
            losses.append(loss)
            optimizer.step()
            optimizer.zero_grad()
        print("Epoch {}, loss = {}".format(epoch + 1, torch.sum(loss) / len(train_loader)))

def test(
    model: any, 
    test_loader: any
) -> any:
    with torch.no_grad():
        losses = []
        accuracies = []
        total_size = 0
        
        for batch in test_loader:
            total_size += len(batch[1])
            loss, corrects = model.test_step(model, batch)
            losses.append(loss)
            accuracies.append(corrects)

        average_loss = np.array(loss).sum() / total_size
        total_accuracy = np.array(accuracies).sum() / total_size
        return average_loss, total_accuracy

def initial_model_training() -> bool:
    GLOBAL_PARAMETERS = current_app.config['GLOBAL_PARAMETERS']

    model_path = 'models/initial_model_parameters.pth'

    if os.path.exists(model_path):
        return False

    torch.manual_seed(GLOBAL_PARAMETERS['seed'])
    
    given_train_loader, given_test_loader = get_train_test_loaders()

    lr_model = FederatedLogisticRegression(dim = GLOBAL_PARAMETERS['input-size'])
    
    train(
        model = lr_model, 
        train_loader = given_train_loader,  
    )
    
    average_loss, total_accuracy = test(
        model = lr_model, 
        test_loader = given_test_loader
    )
    print('Loss:',average_loss)
    print('Accuracy:',total_accuracy)
    parameters = lr_model.get_parameters(lr_model)
    torch.save(parameters, 'models/initial_model_parameters.pth')
    return True

def model_fed_avg(
    updates: any,
    total_sample_size: int    
) -> any:
    weights = []
    biases = []
    for update in updates:
        parameters = update['parameters']
        worker_sample_size = update['samples']
        print(parameters)
        worker_weights = np.array(parameters['weights'][0])
        worker_bias = np.array(parameters['bias'])
        
        adjusted_worker_weights = worker_weights * (worker_sample_size/total_sample_size)
        adjusted_worker_bias = worker_bias * (worker_sample_size/total_sample_size)
        
        weights.append(adjusted_worker_weights.tolist())
        biases.append(adjusted_worker_bias)

    FedAvg_weight = [np.sum(weights,axis = 0)]
    FedAvg_bias = np.sum(biases, axis = 0)

    updated_global_model = OrderedDict([
        ('linear.weight', torch.tensor(FedAvg_weight,dtype=torch.float32)),
        ('linear.bias', torch.tensor(FedAvg_bias,dtype=torch.float32))
    ])
    return updated_global_model

def update_global_model():
    worker_log_path = 'logs/worker_ips.txt' # change to worker status

    if not os.path.exists(worker_log_path):
        return False

    worker_logs = []
    with open(worker_log_path, 'r') as f:
        worker_logs = json.load(f)

    model_folder = 'models'

    usable_updates = []
    files = os.listdir(model_folder)
    current_cycle = 0
    for file in files:
        if 'worker' in file:
            first_split = file.split('.')
            second_split = first_split[0].split('_')
            cycle = int(second_split[2])
            
            if current_cycle < cycle:
                current_cycle = cycle
    
    update_model_path = 'models/global_model_' + str(current_cycle) + '.pth'
    if os.path.exists(update_model_path):
        return True
    
    collective_sample_size = 0
    for file in files:
        if 'worker' in file:
            first_split = file.split('.')
            second_split = first_split[0].split('_')
            worker_id = int(second_split[1])
            cycle = int(second_split[2])
            sample_size = int(second_split[3])

            if cycle == current_cycle:
                if worker_logs[worker_id-1]['status'] == 'complete':
                    print(file)
                    local_model_path = 'models/' + file
                    usable_updates.append({
                        'parameters': torch.load(local_model_path),
                        'samples': sample_size
                    })
                    collective_sample_size = collective_sample_size + sample_size

    new_global_model = model_fed_avg(
        updates = usable_updates,
        total_sample_size = collective_sample_size 
    )
    #print(new_global_model)
    torch.save(new_global_model, update_model_path)
    return True

def evalute_global_model():
    GLOBAL_PARAMETERS = current_app.config['GLOBAL_PARAMETERS']
    model_folder = 'models'
    
    files = os.listdir(model_folder)
    current_cycle = 0
    for file in files:
        if 'global' in file:
            first_split = file.split('.')
            second_split = first_split[0].split('_')
            cycle = int(second_split[2])
            if current_cycle < cycle:
                current_cycle = cycle
    
    global_model_path = 'models/global_model_' + str(current_cycle) + '.pth'
    if not os.path.exists(global_model_path):
        return False
    
    given_parameters = torch.load(global_model_path)
    
    lr_model = FederatedLogisticRegression(dim = GLOBAL_PARAMETERS['input-size'])
    lr_model.apply_parameters(lr_model, given_parameters)

    eval_tensor = torch.load('tensors/evaluation.pt')
    eval_loader = DataLoader(eval_tensor, 64)

    average_loss, total_accuracy = test(
        model = lr_model, 
        test_loader = eval_loader
    )

    print('Loss:',average_loss)
    print('Accuracy:',total_accuracy)

    return True
    



    
            

