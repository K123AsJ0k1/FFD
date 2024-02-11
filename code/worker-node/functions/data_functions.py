from flask import current_app

import numpy as np
import pandas as pd
import torch 
import os
import json

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from collections import OrderedDict

def store_context(
    global_parameters: any,
    worker_parameters: any,
    global_model: any,
    worker_data: any
):
    print('Store context')
    global_parameters_path = 'logs/global_parameters.txt'
    worker_parameters_path = 'logs/worker_parameters.txt'
    
    if not os.path.exists(global_parameters_path):
        with open(global_parameters_path, 'w') as f:
         json.dump(global_parameters, f)

    if not os.path.exists(worker_parameters_path):
        with open(worker_parameters_path, 'w') as f:
         json.dump(worker_parameters, f)

    global_model_path = 'models/global_model_' + str(worker_parameters['cycle']) + '.pth'
    worker_data_path = 'data/used_data_' + str(worker_parameters['cycle']) + '.csv'

    if not os.path.exists(global_model_path):
        weights = global_model['weights']
        bias = global_model['bias']
        formated_parameters = OrderedDict([
            ('linear.weight', torch.tensor(weights,dtype=torch.float32)),
            ('linear.bias', torch.tensor(bias,dtype=torch.float32))
        ])
        #with open(global_model_path, 'w') as f:
         #json.dump(formated_parameters, f)
        torch.save(formated_parameters, global_model_path)
       
    if not os.path.exists(worker_data_path):
       worker_df = pd.DataFrame(worker_data)
       worker_df.to_csv(worker_data_path, index = False)

# Works
def preprocess_into_train_and_test_tensors() -> bool:
    print('Preprocess')
    global_parameters_path = 'logs/global_parameters.txt'
    worker_parameters_path = 'logs/worker_parameters.txt'
    
    if not os.path.exists(global_parameters_path) or not os.path.exists(worker_parameters_path):
       return False
    
    GLOBAL_PARAMETERS = None
    with open(global_parameters_path, 'r') as f:
        GLOBAL_PARAMETERS = json.load(f) 
    WORKER_PARAMETERS = None
    with open(worker_parameters_path, 'r') as f:
        WORKER_PARAMETERS = json.load(f)
    
    worker_data_path = 'data/used_data_' + str(WORKER_PARAMETERS['cycle']) + '.csv'
    
    if not os.path.exists(worker_data_path):
       return False
    
    train_tensor_path = 'tensors/train.pt'
    test_tensor_path = 'tensors/test.pt'
    
    if os.path.exists(train_tensor_path) or os.path.exists(test_tensor_path):
        return False
    

    preprocessed_df = pd.read_csv(worker_data_path)
    print(preprocessed_df)
    preprocessed_df.columns = WORKER_PARAMETERS['columns']
    print(preprocessed_df)
    preprocessed_df = preprocessed_df[GLOBAL_PARAMETERS['used-columns']]
    for column in GLOBAL_PARAMETERS['scaled-columns']:
        mean = preprocessed_df[column].mean()
        std_dev = preprocessed_df[column].std()
        preprocessed_df[column] = (preprocessed_df[column] - mean)/std_dev

    X = preprocessed_df.drop(GLOBAL_PARAMETERS['target-column'], axis = 1).values
    y = preprocessed_df[GLOBAL_PARAMETERS['target-column']].values
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        train_size = WORKER_PARAMETERS['train-test-ratio'], 
        random_state = GLOBAL_PARAMETERS['seed']
    )

    print('Train:',X_train.shape,y_train.shape)
    print('Test:',X_test.shape,y_test.shape)

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)

    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)
    
    train_tensor = TensorDataset(
        torch.tensor(X_train), 
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_tensor = TensorDataset(
        torch.tensor(X_test), 
        torch.tensor(y_test, dtype=torch.float32)
    )
    
    torch.save(train_tensor,train_tensor_path)
    torch.save(test_tensor,test_tensor_path)
    return True