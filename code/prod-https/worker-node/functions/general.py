import os 
import psutil 
import time

from datetime import datetime
from functions.platforms.minio import check_object, create_or_update_object, get_object_data_and_metadata

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
def get_worker_logs():
    storage_folder_path = 'storage'
    central_logs_path = storage_folder_path + '/logs/worker.log'
    logs = None
    with open(central_logs_path, 'r') as f:
        logs = f.readlines()
    return logs
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
        'tensors': worker_folder + '/' + experiments_folder + '/' + str(experiment_name) + '/tensors/replace' ,
        'model': worker_folder + '/' + experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/' + str(cycle) +'/replace',
        'metrics': worker_folder + '/' + experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/' + str(cycle) + '/metrics',
        'resources': worker_folder + '/' + experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/' + str(cycle) + '/resources/replace', # source
        'action-times': worker_folder + '/' + experiments_folder + '/' + str(experiment_name) + '/' + str(experiment) + '/' + str(cycle) + '/times/replace' # area
    }
    return object_paths
# Created
def get_experiments_objects(
    logger: any,
    minio_client: any,
    object: str,
    replacer: str
) -> any:
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

    object_data = None
    object_metadata = None
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
    logger: any,
    minio_client: any,
    object: str,
    replacer: str,
    overwrite: bool,
    object_data: any,
    object_metadata: any
):
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
# Created and works
def get_system_resource_usage():
    net_io_counters = psutil.net_io_counters()
    system_resources = {
        'name': 'system',
        'date': datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
        'time': time.time(),
        'cpu-percent': psutil.cpu_percent(interval = 0.1),
        'ram-percent': psutil.virtual_memory().percent,
        'ram-total-bytes': psutil.virtual_memory().total,
        'ram-free-bytes': psutil.virtual_memory().free,
        'ram-used-bytes': psutil.virtual_memory().used,
        'disk-total-bytes': psutil.disk_usage('.').total,
        'disk-free-bytes': psutil.disk_usage('.').free,
        'disk-used-bytes': psutil.disk_usage('.').used,
        'network-sent-bytes': net_io_counters.bytes_sent,
        'network-received-bytes': net_io_counters.bytes_recv,
        'network-packets-sent': net_io_counters.packets_sent,
        'network-packets-received': net_io_counters.packets_recv,
        'network-packets-sending-errors': net_io_counters.errout,
        'network-packets-reciving-errors': net_io_counters.errin,
        'network-packets-outgoing-dropped': net_io_counters.dropout,
        'network-packets-incoming-dropped': net_io_counters.dropin
    }
    return system_resources
# Created and works
def get_server_resource_usage():
    this_process = psutil.Process(os.getpid())
    cpu_percent = this_process.cpu_percent(interval = 0.1)
    memory_info = this_process.memory_full_info()
    disk_info = this_process.io_counters()
    server_resources = {
        'name': 'server',
        'date': datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'),
        'time': time.time(),
        'cpu-percent': cpu_percent,
        'ram-resident-set-size-bytes': memory_info.rss,
        'ram-virtual-memory-size-bytes': memory_info.vms,
        'ram-shared-memory-bytes': memory_info.shared,
        'ram-code-segment-size-bytes': memory_info.text,
        'ram-data-segment-size-bytes': memory_info.data,
        'ram-library-size-bytes': memory_info.lib,
        'ram-dirty-pages-bytes': memory_info.dirty,
        'ram-unique-set-size-bytes': memory_info.uss,
        'disk-read-bytes': disk_info.read_bytes,
        'disk-write-bytes': disk_info.write_bytes,
        'disk-read-characters-bytes': disk_info.read_chars,
        'disk-write-characters-bytes': disk_info.write_chars
    }
    return server_resources   