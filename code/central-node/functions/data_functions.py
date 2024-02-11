from flask import current_app

import numpy as np
import pandas as pd
import torch 
import os
import pickle

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Works
def central_worker_data_split() -> bool:
    central_pool_path = 'data/Central_Data_Pool.csv'
    worker_pool_path = 'data/Worker_Data_Pool.csv'

    if os.path.exists(central_pool_path) or os.path.exists(worker_pool_path):
        return False
    
    CENTRAL_PARAMETERS = current_app.config['CENTRAL_PARAMETERS']
    WORKER_PARAMETERS = current_app.config['WORKER_PARAMETERS']

    # Code is run in run.py, which is in the root folder
    data_path = 'data/Formated_Fraud_Detection_Data.csv'
    source_df = pd.read_csv(data_path)

    splitted_data_df = source_df.drop('step', axis = 1)
    
    central_data_pool = splitted_data_df.sample(n =  CENTRAL_PARAMETERS['sample-pool'])
    central_indexes = central_data_pool.index.tolist()
    splitted_data_df.drop(central_indexes)
    worker_data_pool = splitted_data_df.sample(n =  WORKER_PARAMETERS['sample-pool'])

    central_data_pool.to_csv('data/Central_Data_Pool.csv', index = False)    
    worker_data_pool.to_csv('data/Worker_Data_Pool.csv', index = False)
    return True
# Works
def preprocess_into_train_test_and_evaluate_tensors() -> bool:
    train_tensor_path = 'tensors/train.pt'
    test_tensor_path = 'tensors/test.pt'
    eval_tensor_path = 'tensors/evaluation.pt'

    if os.path.exists(train_tensor_path) or os.path.exists(test_tensor_path) or os.path.exists(eval_tensor_path):
        return False

    GLOBAL_PARAMETERS = current_app.config['GLOBAL_PARAMETERS']
    CENTRAL_PARAMETERS = current_app.config['CENTRAL_PARAMETERS']
    
    data_path = 'data/Central_Data_Pool.csv'
    central_data_df = pd.read_csv(data_path)
    
    preprocessed_df = central_data_df[GLOBAL_PARAMETERS['used-columns']]
    for column in GLOBAL_PARAMETERS['scaled-columns']:
        mean = preprocessed_df[column].mean()
        std_dev = preprocessed_df[column].std()
        preprocessed_df[column] = (preprocessed_df[column] - mean)/std_dev

    X = preprocessed_df.drop(GLOBAL_PARAMETERS['target-column'], axis = 1).values
    y = preprocessed_df[GLOBAL_PARAMETERS['target-column']].values
        
    X_train_test, X_eval, y_train_test, y_eval = train_test_split(
        X, 
        y, 
        train_size = CENTRAL_PARAMETERS['train-eval-ratio'], 
        random_state = GLOBAL_PARAMETERS['seed']
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_test, 
        y_train_test, 
        train_size = CENTRAL_PARAMETERS['train-test-ratio'], 
        random_state = GLOBAL_PARAMETERS['seed']
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
    
    return True

def split_data_between_workers(
    worker_amount: int
) -> any:
    GLOBAL_PARAMETERS = current_app.config['GLOBAL_PARAMETERS']
    worker_pool_path = 'data/Worker_Data_Pool.csv'

    if not os.path.exists(worker_pool_path):
        return False
    
    worker_pool_df = pd.read_csv(worker_pool_path)
    #print(worker_df)
    #worker_df = worker_pool_df.loc[:,GLOBAL_PARAMETERS['used-columns']]
    #print(worker_df)
    worker_df = worker_pool_df.sample(frac = 1)
    worker_dfs = np.array_split(worker_df, worker_amount)
    
    data_list = []
    index = 1
    for assigned_df in worker_dfs:
        assigned_df.to_csv('data/Worker_' + str(index) + '.csv', index = False)
        #pickled_data = pickle.dumps(assigned_df)
        #pickle_list.append(pickled_data)
        data_list.append(assigned_df.values.tolist())
        index = index + 1

    return data_list, worker_pool_df.columns.tolist()