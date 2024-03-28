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
# Created into one for all getter
def get_central_status(
    logger: any,
    minio_client: any
):
    experiments_folder = 'experiments'
    central_bucket = 'central'
    central_status_path = experiments_folder + '/status'
    central_status_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path
    )
    central_status = central_status_object['data']
    return central_status
# Refactor into one for all setter
def set_central_status(
    logger: any,
    minio_client: any,
    central_status: any
):
    experiments_folder = 'experiments'
    central_bucket = 'central'
    central_status_path = experiments_folder + '/status'
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path,
        data = central_status,
        metadata = {}
    )