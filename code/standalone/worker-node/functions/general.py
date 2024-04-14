import torch 
import os 
import json
import pandas as pd
# Created and works
def get_file_data(
    file_lock: any,
    file_path: str
):
    file_data = None
    if file_path is None:
        return file_data
    storage_folder_path = 'storage'
    used_file_path = storage_folder_path + '/' + file_path
    if not os.path.exists(used_file_path):
        return file_data
    with file_lock:
        if '.txt' in used_file_path:
            with open(used_file_path, 'r') as f:
                file_data = json.load(f)
        if '.csv' in used_file_path:
            file_data = pd.read_csv(used_file_path)
        if '.pt' in used_file_path:
            file_data = torch.load(used_file_path)
    return file_data
# Refactored
def get_files(
    folder_path: str
) -> any:
    storage_folder_path = 'storage'
    checked_directory = storage_folder_path
    if not folder_path == '':
        checked_directory = storage_folder_path + '/' + folder_path
    return os.listdir(checked_directory)
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
    parameter_files = get_files('status')
    highest_experiment_number = 0
    for file in parameter_files:
        if not 'templates' in file:
            experiment_number = int(file.split('_')[1])    
            if highest_experiment_number < experiment_number:
                highest_experiment_number = experiment_number
    return highest_experiment_number
# Refactored
def get_metrics_resources_and_status(
    file_lock: any,
    type: str,
    experiment: int,
    subject: str
) -> any:
    wanted_folder_path = None
    wanted_file_name = None
    if 'worker' in subject:
        wanted_file_name = 'worker'
    if type == 'metrics':
        wanted_folder_path = 'metrics'
        if 'worker' in subject:
            wanted_file_name = 'local'
    if type == 'resources':
        wanted_folder_path = 'resources'
    if type == 'status':
        wanted_folder_path = 'status'
    wanted_data = None
    if wanted_folder_path and wanted_file_name:
        if not experiment == 0:
            wanted_data_path = wanted_folder_path + '/experiment_' + str(experiment) + '/' + wanted_file_name + '.txt'
            wanted_data = get_file_data(
                file_lock = file_lock,
                file_path = wanted_data_path
            )
        else:
            wanted_data = {}
            experiments = get_files(wanted_folder_path)
            for exp in experiments:
                if 'experiment' in exp:
                    exp_id = exp.split('_')[1]
                    data_path = wanted_folder_path + '/' + str(exp) + '/' + wanted_file_name + '.txt' 
                    wanted_data[str(exp_id)] = get_file_data(
                        file_lock = file_lock,
                        file_path = data_path
                    )
    return {'data':wanted_data}
# Refactored
def get_wanted_model(
    file_lock: any,
    experiment: int,
    subject: int,
    cycle: int
) -> any:
    model_folder_path = 'models/experiment_' + str(experiment)
    models = get_files(folder_path = model_folder_path)
    wanted_model_path = None
    for model in models:
        if subject == 'global':
            first_split = model.split('.')
            second_split = first_split[0].split('_')
            if str(cycle) == second_split[1]:
                wanted_model_path = model_folder_path + '/' + model
        if 'local' in subject:
            first_split = model.split('.')
            second_split = first_split[0].split('_')
            if str(cycle) == second_split[1]:
                wanted_model_path = model_folder_path + '/' + model
    
    wanted_model = get_file_data(
        file_lock = file_lock,
        file_path = wanted_model_path
    )
    
    return wanted_model
# Created and works
def get_directory_and_file_sizes() -> any:
    wanted_data = {}
    level_1_directories = get_files(folder_path = '')
    level_0_size = 0
    for level_1_directory in level_1_directories:
        if not level_1_directory in wanted_data:
            wanted_data[level_1_directory] = {}
        level_2_directories = get_files(folder_path = level_1_directory)
        level_1_size = 0
        for level_2_directory in level_2_directories:
            if not level_2_directory in wanted_data:
                wanted_data[level_1_directory][level_2_directory] = {}
            if 'worker.log' == level_2_directory:
                file_path = 'storage/' + level_1_directory + '/' + level_2_directory
                file_size = os.path.getsize(file_path)
                level_1_size += file_size
                wanted_data[level_1_directory][level_2_directory] = file_size # bytes
                continue
            level_2_size = 0
            files = get_files(folder_path = level_1_directory + '/' + level_2_directory)
            for file in files:
                file_path = 'storage/' + level_1_directory + '/' + level_2_directory + '/' + file
                file_size = os.path.getsize(file_path)
                level_2_size += file_size
                wanted_data[level_1_directory][level_2_directory][file] = file_size # bytes
            level_1_size += level_2_size
            wanted_data[level_1_directory][level_2_directory + '-size'] = level_2_size
        level_0_size += level_1_size
        wanted_data[level_1_directory + '-size'] = level_1_size
    wanted_data['storage-size'] = level_0_size
    return {'data':wanted_data}