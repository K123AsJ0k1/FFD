import os

from functions.platforms.minio import check_object, get_object_data_and_metadata, create_or_update_object, get_object_list
from functions.general import format_metadata_dict, encode_metadata_lists_to_strings
# Created and works
def set_object_paths():
    experiments_folder = 'experiments'
    experiment_name = os.environ.get('EXP_NAME')
    experiment = os.environ.get('EXP')
    cycle = os.environ.get('CYCLE')
    next_cycle = ''
    if not cycle is None:
        next_cycle = int(cycle) + 1
    # Has errors during start due to lacking name
    object_paths = {
        'status': experiments_folder + '/status',
        'specifications': experiments_folder + '/specifications',
        'central-template': experiments_folder + '/templates/central-parameters',
        'model-template': experiments_folder + '/templates/model-parameters',
        'worker-template': experiments_folder + '/templates/worker-parameters',
        'experiment-times': experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/times',
        'parameters': experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/parameters/replace',
        'data': experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/data/replace',
        'tensors': experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/tensors/replace' ,
        'global-model': experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/' + str(cycle) +'/global-model',
        'updated-model': experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/' + str(next_cycle) +'/global-model',
        'metrics': experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/' + str(cycle) + '/metrics',
        'workers': experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/' + str(cycle) + '/workers',
        'data-worker': experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/' + str(cycle) + '/data/replace', # worker-id
        'local-models': experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/' + str(cycle) + '/local-models/replace', # worker-id
        'resources': experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/' + str(cycle) + '/resources/replace', # source
        'action-times': experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/' + str(cycle) + '/times/replace' # area
    }
    return object_paths
# Created and works
def get_experiments_objects(
    file_lock: any,
    logger: any,
    minio_client: any,
    object: str,
    replacer: str
) -> any:
    object_data = None
    object_metadata = None
    with file_lock:
        used_bucket = 'central'
        object_paths = set_object_paths()
        
        object_path = object_paths[object]
        if 'replace' in object_path:
            object_path = object_path[:-7] + replacer

        object_exists = check_object(
            logger = logger,
            minio_client = minio_client,
            bucket_name = used_bucket,
            object_path = object_path
        )

        if object_exists:
            fetched_object = get_object_data_and_metadata(
                logger = logger,
                minio_client = minio_client,
                bucket_name = used_bucket,
                object_path = object_path
            )
            object_data = fetched_object['data']
            object_metadata = format_metadata_dict(fetched_object['metadata'])
    return object_data, object_metadata
# Created and works 
def set_experiments_objects(
    file_lock: any,
    logger: any,
    minio_client: any,
    object: str,
    replacer: str,
    overwrite: bool,
    object_data: any,
    object_metadata: any
):
    with file_lock:
        used_bucket = 'central'
        object_paths = set_object_paths()

        object_path = object_paths[object]
        if 'replace' in object_path:
            object_path = object_path[:-7] + replacer

        object_exists = check_object(
            logger = logger,
            minio_client = minio_client,
            bucket_name = used_bucket,
            object_path = object_path
        )
        perform = True
        if object_exists and not overwrite:
            perform = False

        if perform:
            create_or_update_object(
                logger = logger,
                minio_client = minio_client,
                bucket_name = used_bucket,
                object_path = object_path,
                data = object_data,
                metadata = encode_metadata_lists_to_strings(object_metadata)
            )
# Craeted
def get_folder_object_paths(
    file_lock: any,
    logger: any,
    minio_client: any,
    folder_path: str  
):
    formatted_paths = []
    with file_lock:
        used_bucket = 'central'
        folder_objects = get_object_list(
            logger = logger,
            minio_client = minio_client,
            bucket_name = used_bucket,
            path_prefix = folder_path
        )

        for path in folder_objects.keys():
            pkl_split = path.split('.')[0]
            formatted_paths.append(pkl_split)
    return formatted_paths