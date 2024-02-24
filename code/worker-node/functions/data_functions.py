from flask import current_app

import numpy as np
import pandas as pd
import torch  
import os
import json

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

# Refactored
def preprocess_into_train_and_test_tensors(
    logger: any
) -> bool:
    worker_status_path = 'logs/worker_status.txt'
    if not os.path.exists(worker_status_path):
        return False
    
    worker_status = None
    with open(worker_status_path, 'r') as f:
        worker_status = json.load(f)

    if worker_status['completed']:
        return False

    if not worker_status['stored']:
        return False

    if worker_status['preprocessed']:
        return False
    
    worker_data_path = 'data/used_data_' + str(worker_status['cycle']) + '.csv'
    if not os.path.exists(worker_data_path):
       return False
    
    global_parameters_path = 'logs/global_parameters.txt'
    train_tensor_path = 'tensors/train.pt'
    test_tensor_path = 'tensors/test.pt'
    
    os.environ['STATUS'] = 'preprocessing'
    
    GLOBAL_PARAMETERS = None
    with open(global_parameters_path, 'r') as f:
        GLOBAL_PARAMETERS = json.load(f) 

    preprocessed_df = pd.read_csv(worker_data_path)
    preprocessed_df.columns = worker_status['columns']
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
        train_size = worker_status['train-test-ratio'], 
        random_state = GLOBAL_PARAMETERS['seed']
    )

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

    worker_status['preprocessed'] = True
    with open(worker_status_path, 'w') as f:
        json.dump(worker_status, f, indent=4)

    os.environ['STATUS'] = 'preprocessed'

    return True 