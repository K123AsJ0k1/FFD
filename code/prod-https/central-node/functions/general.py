import torch 
import os 
import json
import pandas as pd

from functions.platforms.minio import *

def format_metadata_dict(
    given_metadata: dict
) -> dict:
    # MinIO metadata is first characeter capitalized 
    # and their values are strings due to AMZ format, 
    # which is why the key strings must be made lower
    # and their stirng integers need to be changed to integers 
    fixed_dict = {}
    for key, value in given_metadata.items():
        if value.replace('.','',1).isdigit():
            fixed_dict[key.lower()] = int(value)
        else: 
            fixed_dict[key.lower()] = value
    fixed_dict = decode_metadata_strings_to_lists(fixed_dict)
    return fixed_dict
# Created and works
def encode_metadata_lists_to_strings(
    given_metadata: dict
) -> dict:
    # MinIO metadata only accepts strings and integers, 
    # that have keys without _ characters
    # which is why saving lists in metadata requires
    # making them strings
    modified_dict = {}
    for key,value in given_metadata.items():
        if isinstance(value, list):
            modified_dict[key] = 'list=' + ','.join(map(str, value))
            continue
        modified_dict[key] = value
    return modified_dict 
# Created and works
def decode_metadata_strings_to_lists(
    given_metadata: dict
) -> dict:
    modified_dict = {}
    for key, value in given_metadata.items():
        if isinstance(value, str):
            if 'list=' in value:
                string_integers = value.split('=')[1]
                values = string_integers.split(',')
                if len(values) == 1 and values[0] == '':
                    modified_dict[key] = []
                else:
                    try:
                        modified_dict[key] = list(map(int, values))
                    except:
                        modified_dict[key] = list(map(str, values))
                continue
        modified_dict[key] = value
    return modified_dict

# Refactored and works
def get_central_logs():
    storage_folder_path = 'storage'
    central_logs_path = storage_folder_path + '/logs/central.log'
    logs = None
    with open(central_logs_path, 'r') as f:
        logs = f.readlines()
    return logs
# Refactored and works
def get_metrics_resources_and_status(
    file_lock: any,
    type: str,
    experiment: int,
    subject: str
) -> any:
    wanted_folder_path = None
    wanted_file_name = None
    if 'central' in subject:
        wanted_file_name = 'central'
    if 'workers' in subject:
        wanted_file_name = 'workers'
    if type == 'metrics':
        wanted_folder_path = 'metrics'
        if 'central' in subject:
            wanted_file_name = 'global'
        if 'workers' in subject:
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
            worker_id = str(subject.split('-')[1])
            first_split = model.split('.')
            second_split = first_split[0].split('_')
            if worker_id == second_split[1]:
                if str(cycle) == second_split[2]:
                    wanted_model_path = model_folder_path + '/' + model
    
    wanted_model = get_file_data(
        file_lock = file_lock,
        file_path = wanted_model_path
    )
    
    return wanted_model
# Refactored and works, but could utilize the above or below function
def get_newest_model_updates(
    file_lock:any,
    current_cycle: int
) -> any:
    current_experiment_number = get_current_experiment_number()
    model_folder_path = 'models/experiment_' + str(current_experiment_number)
    files = get_files(folder_path = model_folder_path)
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

                model_parameters = get_file_data(
                    file_lock = file_lock,
                    file_path = local_model_path
                )

                updates.append({
                    'parameters': model_parameters,
                    'samples': sample_size
                })
                collective_sample_size = collective_sample_size + sample_size
    return updates, collective_sample_size
# Refactored and works
def get_models(
    file_lock: any,
    experiment: int,
    subject: str
) -> any:
    wanted_folder_path = 'models'
    wanted_data = None
    wanted_model_name = None
    if 'central' in subject:
        wanted_model_name = 'global'
    if 'workers' in subject:
        wanted_model_name = 'local'
    if wanted_model_name:
        if not experiment == 0:
            wanted_data = {}
            wanted_experiment_path = wanted_folder_path + '/experiment_' + str(experiment) 
            models = get_files(folder_path = wanted_experiment_path)
            for model in models:
                if 'global' == wanted_model_name:
                    first_split = model.split('.')
                    second_split = first_split[0].split('_')
                    wanted_model_path = wanted_experiment_path + '/' + model
                    wanted_model = get_file_data(
                        file_lock = file_lock,
                        file_path = wanted_model_path
                    )

                    data = { 
                        'update-amount': second_split[2],
                        'collective-samples': second_split[3],
                        'weights': wanted_model['linear.weight'].numpy().tolist(),
                        'bias': wanted_model['linear.bias'].numpy().tolist()
                    }
                    wanted_data[str(second_split[1])] = data
                if 'local' == wanted_model_name:  
                    worker_key = subject.split('-')
                    first_split = model.split('.')
                    second_split = first_split[0].split('_')
                    if second_split[1] == str(worker_key):
                        wanted_model_path = wanted_experiment_path + '/' + model
                    
                        wanted_model = get_file_data(
                            file_lock = file_lock,
                            file_path = wanted_model_path
                        )

                        data = { 
                            'train-amount': second_split[3],
                            'weights': wanted_model['linear.weight'].numpy().tolist(),
                            'bias': wanted_model['linear.bias'].numpy().tolist()
                        }
                        wanted_data[str(second_split[1])] = data
        else:
            global_data = {}
            local_data = {}
            experiments = get_files(folder_path = wanted_folder_path)
            for exp in experiments:
                if 'experiment' in exp:
                    exp_id = exp.split('_')[1]
                    if not local_data.get(str(exp_id)):
                        local_data[str(exp_id)] = {}
                    if not global_data.get(str(exp_id)):
                        global_data[str(exp_id)] = {}
                    wanted_experiment_path = wanted_folder_path + '/' + exp 
                    models = get_files(folder_path = wanted_experiment_path)
                    for model in models:
                        first_split = model.split('.')
                        second_split = first_split[0].split('_')
                        if second_split[0] == 'global':
                            wanted_model_path = wanted_experiment_path + '/' + model
                            wanted_model = get_file_data(
                                file_lock = file_lock,
                                file_path = wanted_model_path
                            )

                            data = { 
                                'update-amount': second_split[2],
                                'collective-samples': second_split[3],
                                'weights': wanted_model['linear.weight'].numpy().tolist(),
                                'bias': wanted_model['linear.bias'].numpy().tolist()
                            }
                            global_data[str(exp_id)][str(second_split[1])] = data
                        if second_split[0] == 'local':
                            if not local_data[str(exp_id)].get(str(second_split[1])):
                                local_data[str(exp_id)][str(second_split[1])] = {}

                            wanted_model_path = wanted_experiment_path + '/' + model
                            wanted_model = get_file_data(
                                file_lock = file_lock,
                                file_path = wanted_model_path
                            )
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
            if 'central.log' == level_2_directory:
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
