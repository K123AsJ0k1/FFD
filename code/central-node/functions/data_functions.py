from flask import current_app

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
# Works
def central_worker_data_split():
    CENTRAL_SAMPLE_POOL = current_app.config['CENTRAL_SAMPLE_POOL']
    WORKER_SAMPLE_POOL = current_app.config['WORKER_SAMPLE_POOL']
    # Code is run in run.py, which is in the root folder
    data_path = 'data/Formated_Fraud_Detection_Data.csv'
    source_df = pd.read_csv(data_path)

    splitted_data_df = source_df.drop('step', axis = 1)
    
    central_data_pool = splitted_data_df.sample(n = CENTRAL_SAMPLE_POOL)
    central_indexes = central_data_pool.index.tolist()
    splitted_data_df.drop(central_indexes)
    worker_data_pool = splitted_data_df.sample(n = WORKER_SAMPLE_POOL)

    central_data_pool.to_csv('data/Central_Data_Pool.csv', index = False)    
    worker_data_pool.to_csv('data/Worker_Data_Pool.csv', index = False)
# Works
def preprocess_into_train_test_and_evaluate_tensors():
    GLOBAL_SEED = current_app.config['GLOBAL_SEED']

    SCALED_COLUMNS = current_app.config['SCALED_COLUMNS']
    USED_COLUMNS = current_app.config['USED_COLUMNS']
    TARGET_COLUMN = current_app.config['TARGET_COLUMN']

    CENTRAL_TRAIN_EVALUATION_RATIO = current_app.config['CENTRAL_TRAIN_EVALUATION_RATIO']
    CENTRAL_TRAIN_TEST_RATIO = current_app.config['CENTRAL_TRAIN_TEST_RATIO']

    data_path = 'data/Central_Data_Pool.csv'
    central_data_df = pd.read_csv(data_path)
    
    preprocessed_df = central_data_df[USED_COLUMNS]
    for column in SCALED_COLUMNS:
        mean = preprocessed_df[column].mean()
        std_dev = preprocessed_df[column].std()
        preprocessed_df[column] = (preprocessed_df[column] - mean)/std_dev

    X = preprocessed_df.drop(TARGET_COLUMN, axis = 1).values
    y = preprocessed_df[TARGET_COLUMN].values
        
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

    print('Train:',X_train.shape,y_train.shape)
    print('Test:',X_test.shape,y_test.shape)
    print('Evaluation:',X_eval.shape,y_eval.shape)

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
    current_app.config['GLOBAL_INPUT_SIZE'] = X_train.shape[1]
    