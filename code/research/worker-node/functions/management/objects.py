import os 

from functions.platforms.minio import check_object, create_or_update_object, get_object_data_and_metadata
from functions.general import format_metadata_dict, encode_metadata_lists_to_strings

# Created
def set_object_paths():
    worker_folder = os.environ['WORKER_ID']
    experiments_folder = 'experiments'
    experiment_name = os.environ.get('EXP_NAME')
    experiment = os.environ.get('EXP')
    cycle = os.environ.get('CYCLE')
    object_paths = {
        'status': worker_folder + '/' + experiments_folder + '/status',
        'specifications': worker_folder + '/' + experiments_folder + '/specifications',
        'model-template': worker_folder + '/' + experiments_folder + '/templates/model-parameters',
        'worker-template': worker_folder + '/' + experiments_folder + '/templates/worker-parameters',
        'experiment-times': worker_folder + '/' + experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/times',
        'parameters': worker_folder + '/' + experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/parameters/replace',
        'worker-sample': worker_folder + '/' + experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/worker-sample',
        'tensors': worker_folder + '/' + experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/tensors/replace',
        'model': worker_folder + '/' + experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/' + str(cycle) +'/replace',
        'metrics': worker_folder + '/' + experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/' + str(cycle) + '/metrics',
        'resources': worker_folder + '/' + experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/' + str(cycle) + '/resources/replace', # source
        'action-times': worker_folder + '/' + experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/' + str(cycle) + '/times/replace' # area
    }
    return object_paths
# Created
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
        used_bucket = 'workers'
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
# Created
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
        used_bucket = 'workers'
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