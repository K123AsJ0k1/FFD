from flask import current_app

import torch 
import os 
import json
# Refactored
def get_current_experiment_number():
    parameter_files = os.listdir('storage/parameters')
    highest_experiment_number = 0
    for file in parameter_files:
        if not 'templates' in file:
            experiment_number = int(file.split('_')[1])    
            if highest_experiment_number < experiment_number:
                highest_experiment_number = experiment_number
    return highest_experiment_number
# Refactored
def get_current_global_model() -> any: 
    storage_folder_path = 'storage'
    current_experiment_number = get_current_experiment_number()
    model_folder_path = storage_folder_path + '/models/experiment_' + str(current_experiment_number)
    files = os.listdir(model_folder_path)
    current_global_model = ''
    highest_key = 0
    for file in files:
        first_split = file.split('.')
        second_split = first_split[0].split('_')
        name = str(second_split[0])
        if name == 'global':
            cycle = int(second_split[1])
            if highest_key <= cycle:
                highest_key = cycle
                current_global_model = model_folder_path + '/' + file 
    return torch.load(current_global_model)
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