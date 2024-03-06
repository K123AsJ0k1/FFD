from flask import current_app
import os

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