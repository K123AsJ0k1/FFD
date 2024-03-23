import io
import pickle

def create_bucket(
    logger: any,
    minio_client: any,
    bucket_name: str
) -> bool:
    MINIO_CLIENT = minio_client 
    try:
        MINIO_CLIENT.make_bucket(
            bucket_name = bucket_name
        )
        return True
    except Exception as e:
        logger.error('MinIO bucket creation error')
        logger.error(e)
        return False
    
def check_bucket(
    logger: any,
    minio_client: any,
    bucket_name:str
) -> bool:
    MINIO_CLIENT = minio_client
    try:
        status = MINIO_CLIENT.bucket_exists(bucket_name = bucket_name)
        return status
    except Exception as e:
        logger.error('MinIO bucket checking error')
        logger.error(e)
        return False 
       
def delete_bucket(
    logger: any,
    minio_client: any,
    bucket_name:str
) -> bool:
    MINIO_CLIENT = minio_client
    try:
        MINIO_CLIENT.remove_bucket(
            bucket_name = bucket_name
        )
        return True
    except Exception as e:
        logger.error('MinIO bucket deletion error')
        logger.error(e)
        return False
# Works
def create_object(
    logger: any,
    minio_client: any,
    bucket_name: str, 
    object_path: str, 
    data: any,
    metadata: dict
) -> bool: 
    # Be aware that MinIO objects have a size limit of 1GB, 
    # which might result to large header error
    MINIO_CLIENT = minio_client
    
    pickled_data = pickle.dumps(data)
    length = len(pickled_data)
    buffer = io.BytesIO()
    buffer.write(pickled_data)
    buffer.seek(0)
    try:
        MINIO_CLIENT.put_object(
            bucket_name = bucket_name,
            object_name = object_path + '.pkl',
            data = buffer,
            length = length,
            metadata = metadata
        )
        return True
    except Exception as e:
        logger.error('MinIO object creation error')
        logger.error(e)
        return False
# Works
def check_object(
    logger: any,
    minio_client: any,
    bucket_name: str, 
    object_path: str
) -> bool: 
    MINIO_CLIENT = minio_client
    try:
        object_info = MINIO_CLIENT.stat_object(
            bucket_name = bucket_name,
            object_name = object_path + '.pkl'
        )      
        return True
    except Exception as e:
        return False 
# Works
def delete_object(
    logger: any,
    minio_client: any,
    bucket_name: str, 
    object_path: str
) -> bool: 
    MINIO_CLIENT = minio_client
    try:
        MINIO_CLIENT.remove_object(
            bucket_name = bucket_name, 
            object_name = object_path + '.pkl'
        )
        return True
    except Exception as e:
        logger.error('MinIO object deletion error')
        logger.error(e)
        return False
# Works
def update_object(
    logger: any,
    minio_client: any,
    bucket_name: str, 
    object_path: str, 
    data: any,
    metadata: dict
) -> bool:  
    remove = delete_object(logger,minio_client,bucket_name, object_path)
    if remove:
        create = create_object(logger,minio_client, bucket_name, object_path, data, metadata)
        if create:
            return True
    return False
# works
def create_or_update_object(
    logger: any,
    minio_client: any,
    bucket_name: str, 
    object_path: str, 
    data: any, 
    metadata: dict
) -> any:
    bucket_status = check_bucket(logger,minio_client,bucket_name)
    if not bucket_status:
        creation_status = create_bucket(logger,minio_client,bucket_name)
        if not creation_status:
            return None
    object_status = check_object(logger,minio_client,bucket_name, object_path)
    if not object_status:
        return create_object(logger,minio_client,bucket_name, object_path, data, metadata)
    else:
        return update_object(logger,minio_client,bucket_name, object_path, data, metadata)
# Works
def get_object_data_and_metadata(
    logger: any,
    minio_client: any,
    bucket_name: str, 
    object_path: str
) -> dict:
    MINIO_CLIENT = minio_client
    
    try:
        given_object_info = MINIO_CLIENT.stat_object(
            bucket_name = bucket_name, 
            object_name = object_path + '.pkl'
        )
        # There seems to be some kind of a limit
        # with the amount of request a client 
        # can make, which is why this variable
        # is set here to give more time got the client
        # to complete the request
        given_metadata = given_object_info.metadata
        
        given_object_data = MINIO_CLIENT.get_object(
            bucket_name = bucket_name, 
            object_name = object_path + '.pkl'
        )
        given_pickled_data = given_object_data.data
        
        try:
            given_data = pickle.loads(given_pickled_data)
            relevant_metadata = {} 
            for key, value in given_metadata.items():
                if 'x-amz-meta' in key:
                    key_name = key[11:]
                    relevant_metadata[key_name] = value
            return {'data': given_data, 'metadata': relevant_metadata}
        except Exception as e:
            logger.error('MinIO object pickle decoding error')
            logger.error(e)
            return None 
    except Exception as e:
        logger.error('MinIO object fetching error')
        logger.error(e)
        return None
# Works
def get_object_list(
    logger: any,
    minio_client: any,
    bucket_name: str,
    path_prefix: str
) -> dict:
    MINIO_CLIENT = minio_client
    try:
        objects = MINIO_CLIENT.list_objects(bucket_name = bucket_name, prefix = path_prefix, recursive = True)
        object_dict = {}
        for obj in objects:
            object_name = obj.object_name
            object_info = MINIO_CLIENT.stat_object(
                bucket_name = bucket_name,
                object_name = object_name
            )
            given_metadata = {} 
            for key, value in object_info.metadata.items():
                if 'X-Amz-Meta' in key:
                    key_name = key[11:]
                    given_metadata[key_name] = value
            object_dict[obj.object_name] = given_metadata
        return object_dict
    except Exception as e:
        return None  
# Works
def delete_objects(
    logger: any,
    minio_client: any, 
    bucket_name: str,
    path_prefix: str
):
    objects = get_object_list(logger,minio_client,bucket_name, path_prefix)
    for object_name in objects.keys():
        pkl_split = object_name.split('.')[0]
        delete_object(logger,minio_client,bucket_name,pkl_split)