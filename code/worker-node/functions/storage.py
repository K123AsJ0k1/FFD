from flask import current_app

import pandas as pd
import torch  
import os
import json

from collections import OrderedDict
 
# Refactor
def store_training_context(
    parameters: any,
    global_model: any,
    df_data: list,
    df_columns: list
) -> any:
    # Separate training artifacts will have the following folder format of experiment_(int)
    current_experiment_number = get_current_experiment_number()
    worker_status_path = 'status/experiment_' + str(current_experiment_number) + '/worker.txt'
    if not os.path.exists(worker_status_path):
        return 'no status'
    
    worker_status = None
    with open(worker_status_path, 'r') as f:
        worker_status = json.load(f)
    
    if worker_status['complete']:
        return 'complete'
    
    if not parameters['id'] == worker_status['id']:
        return 'wrong id'
    
    if worker_status['stored'] and not worker_status['updated']:
        return 'ongoing jobs'
    
    if parameters['model'] == None:
        worker_status['completed'] = True
        worker_status['cycle'] = parameters['cycle']
    else:
        parameters_folder_path = 'parameters/experiment_' + str(current_experiment_number)
    
        os.makedirs(parameters_folder_path,exist_ok=True)

        model_parameters_path = parameters_folder_path + '/model.txt'
        worker_parameters_path = parameters_folder_path + '/worker.txt'

        with open(model_parameters_path, 'w') as f:
            json.dump(parameters['model'], f, indent=4)

        with open(worker_parameters_path, 'w') as f:
            json.dump(parameters['worker'], f, indent=4)

        worker_status['preprocessed'] = False
        worker_status['trained'] = False
        worker_status['updated'] = False
        worker_status['completed'] = False
        worker_status['cycle'] = parameters['cycle']

    os.environ['STATUS'] = 'storing'
    
    model_folder_path = 'models/experiment_' + str(current_experiment_number)
    global_model_path = model_folder_path + '/global_' + str(worker_status['cycle']-1) + '.pth'
    
    weights = global_model['weights']
    bias = global_model['bias']
    
    formated_parameters = OrderedDict([
        ('linear.weight', torch.tensor(weights,dtype=torch.float32)),
        ('linear.bias', torch.tensor(bias,dtype=torch.float32))
    ])
    
    torch.save(formated_parameters, global_model_path)
    if not df_data == None:
        worker_data_path = 'data/sample_' + str(worker_status['cycle']) + '.csv'
        worker_df = pd.DataFrame(df_data, columns = df_columns)
        worker_df.to_csv(worker_data_path, index = False)
        worker_status['preprocessed'] = False

    worker_status['stored'] = True
    with open(worker_status_path, 'w') as f:
        json.dump(worker_status, f, indent=4)

    os.environ['STATUS'] = 'stored'

    return 'stored'
# Refactor
def store_local_metrics(
   metrics: any
) -> bool:
    #worker_status_path = 'logs/worker_status.txt'
    #if not os.path.exists(worker_status_path):
    #    return False
    #worker_status = None
    #with open(worker_status_path, 'r') as f:
    #    worker_status = json.load(f)

    new_key = len(worker_status['local-metrics'])
    worker_status['local-metrics'][str(new_key)] = metrics
    with open(worker_status_path, 'w') as f:
        json.dump(worker_status, f, indent=4) 
    return True