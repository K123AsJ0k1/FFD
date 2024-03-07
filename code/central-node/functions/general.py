from flask import current_app

import torch 
import os 
import json
# Refactored and works
def get_central_logs():
    storage_folder_path = 'storage'
    central_logs_path = storage_folder_path + '/central.log'
    logs = None
    with open(central_logs_path, 'r') as f:
        logs = f.readlines()
    return logs
# Refactored and works
def get_current_experiment_number():
    parameter_files = os.listdir('storage/parameters')
    highest_experiment_number = 0
    for file in parameter_files:
        if not 'templates' in file:
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
# Refactored and works
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
        if 'local' in subject:
            worker_id = str(subject.split('-')[1])
            first_split = model.split('.')
            second_split = first_split[0].split('_')
            if worker_id == second_split[1]:
                if str(cycle) == second_split[2]:
                    wanted_model = model_folder_path + '/' + model
    return torch.load(wanted_model)
# Created and works
def get_newest_model_updates(
    current_cycle: int
) -> any:
    storage_folder_path = 'storage'
    current_experiment_number = get_current_experiment_number()
    model_folder_path = storage_folder_path + '/models/experiment_' + str(current_experiment_number)
    files = os.listdir(model_folder_path)
    updates = []
    collective_sample_size = 0
    for file in files:
        if 'local' in file:
            first_split = file.split('.')
            second_split = first_split[0].split('_')
            cycle = int(second_split[2])
            sample_size = int(second_split[3])
            if cycle == current_cycle:
                local_model_path = model_folder_path + '/' + file
                updates.append({
                    'parameters': torch.load(local_model_path),
                    'samples': sample_size
                })
                collective_sample_size = collective_sample_size + sample_size
    return updates, collective_sample_size
# Refactored and works
def get_models(
    experiment: int,
    subject: str
) -> any:
    storage_folder_path = 'storage'
    wanted_folder_path = storage_folder_path + '/models'
    wanted_data = None
    if not experiment == 0:
        wanted_data = {}
        wanted_experiment_path = wanted_folder_path + '/experiment_' + str(experiment) 
        models = os.listdir(wanted_experiment_path)
        for model in models:
            if 'global' == subject:
                first_split = model.split('.')
                second_split = first_split[0].split('_')
                wanted_model_path = wanted_experiment_path + '/' + model
                wanted_model = torch.load(wanted_model_path)
                data = { 
                    'update-amount': second_split[2],
                    'collective-samples': second_split[3],
                    'weights': wanted_model['linear.weight'].numpy().tolist(),
                    'bias': wanted_model['linear.bias'].numpy().tolist()
                }
                wanted_data[str(second_split[1])] = data
            if 'local' in subject:  
                worker_key = subject.split('-')
                first_split = model.split('.')
                second_split = first_split[0].split('_')
                if second_split[1] == str(worker_key):
                    wanted_model_path = wanted_experiment_path + '/' + model
                    wanted_model = torch.load(wanted_model_path)
                    data = { 
                        'train-amount': second_split[3],
                        'weights': wanted_model['linear.weight'].numpy().tolist(),
                        'bias': wanted_model['linear.bias'].numpy().tolist()
                    }
                    wanted_data[str(second_split[1])] = data
    else:
        global_data = {}
        local_data = {}
        experiments = os.listdir(wanted_folder_path)
        for exp in experiments:
            if 'experiment' in exp:
                exp_id = exp.split('_')[1]
                if not local_data.get(str(exp_id)):
                    local_data[str(exp_id)] = {}
                if not global_data.get(str(exp_id)):
                    global_data[str(exp_id)] = {}
                wanted_experiment_path = wanted_folder_path + '/' + exp 
                models = os.listdir(wanted_experiment_path)
                for model in models:
                    if 'global' in model:
                        first_split = model.split('.')
                        second_split = first_split[0].split('_')
                        wanted_model_path = wanted_experiment_path + '/' + model
                        wanted_model = torch.load(wanted_model_path)
                        data = { 
                            'update-amount': second_split[2],
                            'collective-samples': second_split[3],
                            'weights': wanted_model['linear.weight'].numpy().tolist(),
                            'bias': wanted_model['linear.bias'].numpy().tolist()
                        }
                        global_data[str(exp_id)][str(second_split[1])] = data
                    if 'local' in model:
                        first_split = model.split('.')
                        second_split = first_split[0].split('_')
                        if not local_data[str(exp_id)].get(str(second_split[1])):
                            local_data[str(exp_id)][str(second_split[1])] = {}

                        wanted_model_path = wanted_experiment_path + '/' + model
                        wanted_model = torch.load(wanted_model_path)
                        data = { 
                            'train-amount': second_split[3],
                            'weights': wanted_model['linear.weight'].numpy().tolist(),
                            'bias': wanted_model['linear.bias'].numpy().tolist()
                        }
                        local_data[str(exp_id)][str(second_split[1])][str(second_split[2])] = data
        wanted_data = {
            'global': global_data,
            'local': local_data
        }
    return {'data':wanted_data}