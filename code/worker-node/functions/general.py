from flask import current_app
import os
import json
import torch

# Refactored and works
def get_worker_logs():
    storage_folder_path = 'storage'
    central_logs_path = storage_folder_path + '/logs/worker.log'
    logs = None
    with open(central_logs_path, 'r') as f:
        logs = f.readlines()
    return logs
# Refactored and works
def get_current_experiment_number():
    parameter_files = os.listdir('storage/status')
    highest_experiment_number = 0
    for file in parameter_files:
        if not 'template' in file:
            experiment_number = int(file.split('_')[1])    
            if highest_experiment_number < experiment_number:
                highest_experiment_number = experiment_number
    return highest_experiment_number
# Refactored and works
def get_metrics_resources_and_status(
    type: str,
    experiment: int,
    subject: str
) -> any:
    storage_folder_path = 'storage'
    wanted_folder_path = None
    if type == 'metrics':
        wanted_folder_path = storage_folder_path + '/metrics'
    if type == 'resources':
        wanted_folder_path = storage_folder_path + '/resources'
    if type == 'status':
        wanted_folder_path = storage_folder_path + '/status'
    wanted_data = None
    if not experiment == 0:
        wanted_data_path = wanted_folder_path + '/experiment_' + str(experiment) + '/' + subject + '.txt'
        with open(wanted_data_path, 'r') as f:
            wanted_data = json.load(f)
    else:
        wanted_data = {}
        experiments = os.listdir(wanted_folder_path)
        for exp in experiments:
            if 'experiment' in exp:
                exp_id = exp.split('_')[1]
                data_path = wanted_folder_path + '/' + str(exp) + '/' + subject + '.txt' 
                with open(data_path, 'r') as f:
                    wanted_data[str(exp_id)] = json.load(f)
    return {'data':wanted_data}
# Created and works
def get_wanted_model(
    experiment: int,
    subject: int,
    cycle: int
) -> any:
    storage_folder_path = 'storage'
    model_folder_path = storage_folder_path + '/models/experiment_' + str(experiment)
    models = os.listdir(model_folder_path)
    wanted_model = None
    for model in models:
        if subject == 'global':
            first_split = model.split('.')
            second_split = first_split[0].split('_')
            if str(cycle) == second_split[1]:
                wanted_model = model_folder_path + '/' + model
        if subject == 'local':
            first_split = model.split('.')
            second_split = first_split[0].split('_')
            if str(cycle) == second_split[1]:
                wanted_model = model_folder_path + '/' + model
    return torch.load(wanted_model)