from flask import current_app
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

import requests 

from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from functions.data_functions import *
# Refactored
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
    def get_parameters(model):
        return model.state_dict()

    @staticmethod
    def apply_parameters(model, parameters):
        model.load_state_dict(parameters)
# Refactored
def get_train_test_loaders_loaders() -> any:
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
# Refactored
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
# Refactored
def test(
    model: any, 
    test_loader: any
) -> any:
    with torch.no_grad():
        #losses = []
        total_size = 0
        total_confusion_matrix = [0,0,0,0]
        
        for batch in test_loader:
            total_size += len(batch[1])
            _, correct = batch
            _, preds = model.test_step(model, batch)
            #losses.append(loss)
            
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
 
        #average_loss = np.array(loss).sum() / total_size
        # 'loss': float(round(average_loss,5)),
        TP, FP, TN, FN = total_confusion_matrix

        TPR = TP/(TP+FN)
        TNR = TN/(TN+FP)
        PPV = TP/(TP+FP)
        FNR = FN/(FN+TP)
        FPR = FP/(FP+TN)
        BA = (TPR+TNR)/2
        ACC = (TP + TN)/(TP + TN + FP + FN)
        
        metrics = {
            'confusion': total_confusion_matrix,
            'recall': float(round(TPR,5)),
            'selectivity': float(round(TNR,5)),
            'precision': float(round(PPV,5)),
            'miss-rate': float(round(FNR,5)),
            'fall-out': float(round(FPR,5)),
            'balanced-accuracy': float(round(BA,5)),
            'accuracy': float(round(ACC,5))
        }
        
        return metrics
# Refactored
def local_model_training() -> any:
    global_parameters_path = 'logs/global_parameters.txt'
    worker_parameters_path = 'logs/worker_parameters.txt'

    if not os.path.exists(global_parameters_path):
        return False

    if not os.path.exists(worker_parameters_path):
        return False
 
    GLOBAL_PARAMETERS = None
    with open(global_parameters_path, 'r') as f:
        GLOBAL_PARAMETERS = json.load(f)
    
    WORKER_PARAMETERS = None
    with open(worker_parameters_path, 'r') as f:
        WORKER_PARAMETERS = json.load(f)
    
    global_model_path = 'models/global_model_' + str(WORKER_PARAMETERS['cycle']) + '.pth'
    local_model_path = 'models/local_model_' + str(WORKER_PARAMETERS['cycle']) + '.pth'

    if os.path.exists(global_model_path):
        return False
    
    if os.path.exists(local_model_path):
        return False
    
    os.environ['STATUS'] = 'training'

    given_parameters = torch.load(global_model_path)

    torch.manual_seed(GLOBAL_PARAMETERS['seed'])
    
    given_train_loader, given_test_loader = get_train_test_loaders_loaders()

    lr_model = FederatedLogisticRegression(dim = GLOBAL_PARAMETERS['input-size'])
    lr_model.apply_parameters(lr_model, given_parameters)

    train(
        model = lr_model, 
        train_loader = given_train_loader,  
    )
    
    test_metrics = test(
        model = lr_model, 
        test_loader = given_test_loader
    )

    store_local_metrics(
        metrics = test_metrics
    )
    
    parameters = lr_model.get_parameters(lr_model)
    torch.save(parameters, local_model_path)
    return True
# Created
def run_training_pipeline(
    logger: any
):
    status = preprocess_into_train_and_test_tensors()
    status = local_model_training()
# Created    
def send_update(
    logger: any, 
    central_address: str
):  
    if os.environ.get('STATUS') == 'updated':
        return False

    worker_parameters_path = 'logs/worker_parameters.txt'
    if not os.path.exists(worker_parameters_path):
        return False

    WORKER_PARAMETERS = None
    with open(worker_parameters_path, 'r') as f:
        WORKER_PARAMETERS = json.load(f)

    os.environ['STATUS'] = 'updating'

    local_model_path = 'models/local_model_' + str(WORKER_PARAMETERS['cycle']) + '.pth'
    local_model = torch.load(local_model_path)

    formatted_local_model = {
      'weights': local_model['linear.weight'].numpy().tolist(),
      'bias': local_model['linear.bias'].numpy().tolist()
    }

    train_tensor = torch.load('tensors/train.pt')
    
    payload = {
        'worker-id': int(WORKER_PARAMETERS['worker-id']),
        'local-model': formatted_local_model,
        'cycle': WORKER_PARAMETERS['cycle'],
        'train-size': len(train_tensor)
    }
    
    json_payload = json.dumps(payload)
    central_url = central_address + '/update'
    try:
        response = requests.post(
            url = central_url, 
            json = json_payload,
            headers = {
                'Content-type':'application/json', 
                'Accept':'application/json'
            }
        )
        if response.status_code == 200:
            os.environ['STATUS'] = 'updated'
    except Exception as e:
        logger.error('Status sending error:', e)