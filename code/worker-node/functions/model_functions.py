from flask import current_app
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import requests

from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

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

def get_loaders() -> any:
    global_parameters_path = 'logs/global_parameters.txt'
    GLOBAL_PARAMETERS = None
    with open(global_parameters_path, 'r') as f:
        GLOBAL_PARAMETERS = json.load(f)
    
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
    global_parameters_path = 'logs/global_parameters.txt'
    GLOBAL_PARAMETERS = None
    with open(global_parameters_path, 'r') as f:
        GLOBAL_PARAMETERS = json.load(f)
    
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
# Works
def local_model_training(
    cycle: int
) -> any:
    print('Local model training')
    global_parameters_path = 'logs/global_parameters.txt'
    global_model_path = 'models/global_model_' + str(cycle) + '.pth'
    local_model_path = 'models/local_model_' + str(cycle) + '.pth'

    if not os.path.exists(global_parameters_path) or not os.path.exists(global_model_path):
        return False
    
    if os.path.exists(local_model_path):
        return False

    GLOBAL_PARAMETERS = None
    with open(global_parameters_path, 'r') as f:
        GLOBAL_PARAMETERS = json.load(f)

    given_parameters = torch.load(global_model_path)

    model_path = 'models/worker_model_parameters' + str(cycle) + '.pth'
    if os.path.exists(model_path):
        return False
 
    torch.manual_seed(GLOBAL_PARAMETERS['seed'])
    
    given_train_loader, given_test_loader = get_loaders()

    lr_model = FederatedLogisticRegression(dim = GLOBAL_PARAMETERS['input-size'])
    lr_model.apply_parameters(lr_model, given_parameters)

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
    torch.save(parameters, local_model_path)
    return True

def send_update(
    logger:any, 
    central_address:str
):  
    logger.warning('Send update')
    model_folder = 'models'
    files = os.listdir(model_folder)
    cycle = 0
    for file in files:
        if 'global_model' in file:
            first_split = file.split('_')
            second_split = first_split[-1].split('.')
            file_cycle = int(second_split[0])
            if cycle < file_cycle:
                cycle = file_cycle
    training_status = local_model_training(
        cycle = cycle
    )

    worker_parameters_path = 'logs/worker_parameters.txt'

    WORKER_PARAMETERS = None
    with open(worker_parameters_path, 'r') as f:
        WORKER_PARAMETERS = json.load(f)
    
    local_model_path = 'models/local_model_' + str(cycle) + '.pth'
    local_model = torch.load(local_model_path)

    formatted_local_model = {
      'weights': local_model['linear.weight'].numpy().tolist(),
      'bias': local_model['linear.bias'].numpy().tolist()
    }

    train_tensor = torch.load('tensors/train.pt')
    


    payload = {
        'worker-id': ,
        'local-model': formatted_local_model,
        'cycle': cycle,
        'train-size': len(train_tensor)
    }

    address = central_address + '/update'
    try:
        response = requests.post(
            url = address
        )
        logger.warning(response.status_code)
    except Exception as e:
        logger.error('Registration error')
        logger.error(e) 