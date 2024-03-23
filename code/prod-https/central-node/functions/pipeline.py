from functions.platforms.minio import get_object_data_and_metadata, create_or_update_object
from functions.general import format_metadata_dict, encode_metadata_lists_to_strings
from datetime import datetime
# Refactored and works
def start_pipeline(
    file_lock: any,
    logger: any,
    minio_client: any,
    parameters: any,
    df_data: list,
    df_columns: list
):  
    central_bucket = 'central'
    central_status_path = 'experiments/status'
    central_status_object = get_object_data_and_metadata(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path
    )
    central_status = central_status_object['data']

    if central_status is None:
        return False
    
    if central_status['start'] and not central_status['complete']:
        return False
    
    times = {
        'experiment-date': datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
        'experiment-time-start':0,
        'experiment-time-end':0,
        'experiment-total-seconds': 0,
    }
    object_path = 'experiments/' + str(central_status['experiment']) + '/times'
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = object_path,
        data = times,
        metadata = {}
    )
    
    for name in parameters.keys():
        template_parameter_path = 'experiments/templates' + '/' + str(name) + '-parameters'
        template_parameter_object = get_object_data_and_metadata(
            logger = logger,
            minio_client = minio_client,
            bucket_name = central_bucket ,
            object_path = template_parameter_path
        )
        template_parameters = template_parameter_object['data']
        given_parameters = parameters[name]
        for key in template_parameters.keys():
            template_parameters[key] = given_parameters[key]
        
        modified_parameter_path = 'experiments/' + str(central_status['experiment']) + '/parameters/' + str(name)
        create_or_update_object(
            logger = logger,
            minio_client = minio_client,
            bucket_name = central_bucket,
            object_path = modified_parameter_path,
            data = template_parameters,
            metadata = {}
        )
    formatted_metadata = encode_metadata_lists_to_strings({'columns': df_columns})
    source_data_path = 'experiments/' + str(central_status['experiment']) + '/data/source'
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = source_data_path,
        data = df_data,
        metadata = formatted_metadata
    )
    
    central_status['start'] = True
    create_or_update_object(
        logger = logger,
        minio_client = minio_client,
        bucket_name = central_bucket,
        object_path = central_status_path,
        data = central_status,
        metadata = {}
    )
    
    return True
# Refactor
def processing_pipeline(
    task_file_lock: any,
    task_logger: any
):
    # 
    status = central_worker_data_split(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Central-worker data split:' + str(status))
    # 
    status = preprocess_into_train_test_and_evaluate_tensors(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Central pool preprocessing:' + str(status))
'''
# Refactor
def model_pipeline(
    task_file_lock: any,
    task_logger: any
):  
    # 
    status = initial_model_training(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Initial model training:' + str(status))
# Refactor
def update_pipeline(
    task_file_lock: any,
    task_logger: any
):
    # 
    status = split_data_between_workers(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Worker data split:' + str(status))
    # 
    status = send_context_to_workers(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Worker context sending:' + str(status))
# Refactor
def aggregation_pipeline(
    task_file_lock: any,
    task_logger: any
):
    # 
    status = update_global_model(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Updating global model:' + str(status))
    # 
    status = evalute_global_model(
        file_lock = task_file_lock,
        logger = task_logger
    )
    task_logger.info('Global model evaluation:' + str(status))
'''
