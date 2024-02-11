from flask import current_app

import numpy as np
import pandas as pd
import torch 
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset



# Works
def preprocess_into_train_and_test_tensors() -> bool:
    train_tensor_path = 'tensors/train.pt'
    test_tensor_path = 'tensors/test.pt'

    if os.path.exists(train_tensor_path) or os.path.exists(test_tensor_path):
        return False

    GLOBAL_SEED = current_app.config['GLOBAL_SEED']

    GLOBAL_SCALED_COLUMNS = current_app.config['GLOBAL_SCALED_COLUMNS']
    GLOBAL_USED_COLUMNS = current_app.config['GLOBAL_USED_COLUMNS']
    GLOBAL_TARGET_COLUMN = current_app.config['GLOBAL_TARGET_COLUMN']

    CENTRAL_TRAIN_TEST_RATIO = current_app.config['CENTRAL_TRAIN_TEST_RATIO']

    data_path = 'data/Central_Data_Pool.csv'
    central_data_df = pd.read_csv(data_path)
    
    preprocessed_df = central_data_df[GLOBAL_USED_COLUMNS]
    for column in GLOBAL_SCALED_COLUMNS:
        mean = preprocessed_df[column].mean()
        std_dev = preprocessed_df[column].std()
        preprocessed_df[column] = (preprocessed_df[column] - mean)/std_dev

    X = preprocessed_df.drop(GLOBAL_TARGET_COLUMN, axis = 1).values
    y = preprocessed_df[GLOBAL_TARGET_COLUMN].values
        
    X_train_test, X_eval, y_train_test, y_eval = train_test_split(
        X, 
        y, 
        train_size = CENTRAL_TRAIN_EVALUATION_RATIO, 
        random_state = GLOBAL_SEED
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_test, 
        y_train_test, 
        train_size = CENTRAL_TRAIN_TEST_RATIO, 
        random_state = GLOBAL_SEED
    )

    #print('Train:',X_train.shape,y_train.shape)
    #print('Test:',X_test.shape,y_test.shape)
    #print('Evaluation:',X_eval.shape,y_eval.shape)

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    X_eval = np.array(X_eval, dtype=np.float32)

    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)
    y_eval = np.array(y_eval, dtype=np.int32)
    
    train_tensor = TensorDataset(
        torch.tensor(X_train), 
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_tensor = TensorDataset(
        torch.tensor(X_test), 
        torch.tensor(y_test, dtype=torch.float32)
    )
    eval_tensor = TensorDataset(
        torch.tensor(X_eval), 
        torch.tensor(y_eval, dtype=torch.float32)
    )

    torch.save(train_tensor,'tensors/train.pt')
    torch.save(test_tensor,'tensors/test.pt')
    torch.save(eval_tensor,'tensors/evaluation.pt')
    #current_app.config['GLOBAL_INPUT_SIZE'] = X_train.shape[1]
    return True