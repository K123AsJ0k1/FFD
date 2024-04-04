from flask import current_app

'''
MLFlow experiment format:
- list_experiments(): list
    - experiment: dict example
        artifact_location: artifact_location='s3://mlflow/0',
        creation_time: None,
        experiment_id: '0',
        last_update_time: None,
        lifecycle_stage: 'active',
        name: 'default',
        tags: {}
MLFlow run format:
- run: dict example
    - data: dict
        - metrics: {}
        - params: {}
        - tags: {}
    - info: dict
        - artifact_uri: 's3://mlflow/12/ebfd30916f62428c885d38b2ce721db2/artifacts'
        - end_time: None
        - experiment_id: '12'
        - lifecycle_stage: 'active'
        - run_id: 'ebfd30916f62428c885d38b2ce721db2'
        - run_uuid: 'ebfd30916f62428c885d38b2ce721db2'
        - start_time: 1699256493235
        - status: 'RUNNING'
        - user_id: 'unknown'
'''
# Refactored and works
def start_experiment(
    logger: any,
    mlflow_client: any,
    experiment_name: str,
    experiment_tags: dict
) -> int:
    MLFLOW_CLIENT = mlflow_client
    try:
        experiment_id = MLFLOW_CLIENT.create_experiment(
            name = experiment_name,
            tags = experiment_tags,
            artifact_location="s3://mlflow/mlruns"
        )
        return experiment_id
    except Exception as e:
        logger.error('MLflow experiment starting error')
        logger.error(e)
        return None
# Refactored
def check_experiment(
    logger: any,
    mlflow_client: any,
    experiment_name: str
) -> dict:
    MLFLOW_CLIENT = mlflow_client
    try:
        experiment_object = MLFLOW_CLIENT.get_experiment_by_name(
            name = experiment_name
        )
        return experiment_object
    except Exception as e:
        logger.error('MLflow experiment checking error')
        logger.error(e)
        return None
# Refactored
def get_experiments(
    logger: any,
    mlflow_client: any,
    experiment_type: int,
    max_amount: int,
    filter: str
) -> dict:
    # Types are ACTIVE_ONLY = 1, DELETED_ONLY = 2 and ALL = 3
    MLFLOW_CLIENT = mlflow_client
    try:
        experiment_objects = MLFLOW_CLIENT.search_experiments(
            view_type = experiment_type,
            max_results = max_amount,
            filter_string = filter
        )
        experiment_dict = {}
        for experiment in experiment_objects:
            experiment_dict[experiment.name] = {
                'id': experiment.experiment_id,
                'stage': experiment.lifecycle_stage,
                'tags': experiment.tags,
                'location': experiment.artifact_location,
                'created': experiment.creation_time,
                'updated': experiment.last_update_time
            }
        return experiment_dict
    except Exception as e:
        logger.error('MLflow experiment getting error')
        logger.error(e)
        return None
# Refactored
def start_run(
    logger: any,
    mlflow_client: any,
    experiment_id: str,
    tags: dict,
    name: str
) -> dict:
    MLFLOW_CLIENT = mlflow_client 
    try:
        run_object = MLFLOW_CLIENT.create_run(
            experiment_id = experiment_id,
            tags = tags,
            run_name = name
        )
        run_dict = {
            'e_id': run_object.info.experiment_id,
            'id': run_object.info.run_id,
            'name': run_object.info.run_name,
            'stage': run_object.info.lifecycle_stage,
            'status': run_object.info.status
        }
        return run_dict
    except Exception as e:
        logger.error('MLflow run starting error')
        logger.error(e)
        return None
# Refactored
def check_run(
    logger: any,
    mlflow_client: any,
    run_id: str
) -> dict:
    MLFLOW_CLIENT = mlflow_client
    try:
        run_object = MLFLOW_CLIENT.get_run(
            run_id = run_id
        )
        current_app.logger.info('Checking succeeded')
        run_dict = {
            'e_id': run_object.info.experiment_id,
            'id': run_object.info.run_id,
            'name': run_object.info.run_name,
            'stage': run_object.info.lifecycle_stage,
            'status': run_object.info.status,
            'parameters': run_object.data.params,
            'metrics': run_object.data.metrics,
            'start_time': run_object.info.start_time,
            'end_time': run_object.info.end_time
        }
        return run_dict
    except Exception as e:
        logger.error('MLflow run checking error')
        logger.error(e)
        return None
# Refactored
def update_run(
    logger: any,
    mlflow_client: any,
    run_id: str,
    parameters: dict,
    metrics: dict,
    artifacts: dict
) -> bool:
    MLFLOW_CLIENT = mlflow_client 
    try:
        for param_key, param_value in parameters.items():
            MLFLOW_CLIENT.log_param(
                run_id = run_id,
                key = param_key,
                value = param_value
            )
        for metric_key,metric_value in metrics.items():
            MLFLOW_CLIENT.log_metric(
                run_id = run_id,
                key = metric_key,
                value = metric_value
            )
        for path in artifacts:
            MLFLOW_CLIENT.log_artifact(
                run_id = run_id,
                local_path = path
            )
        return True
    except Exception as e:
        logger.error('MLflow run updating error')
        logger.error(e)
        return False
# Recatored
def end_run(
    logger: any,
    mlflow_client: any,
    run_id: str,
    status: str
) -> bool:
    # run status are FAILED = 4, FINISHED = 3, KILLED = 5, RUNNING = 1 and SCHEDULED = 2
    MLFLOW_CLIENT = mlflow_client
    try:
        MLFLOW_CLIENT.set_terminated(
            run_id = run_id,
            status = status
        )
        return True
    except Exception as e:
        logger.error('MLflow run ending error')
        logger.error(e)
        return False
# Refactored
def get_runs(
    logger: any,
    mlflow_client: any,
    experiment_ids: list,
    filter: str,
    type: int,
    max_amount: int
) -> dict:
    MLFLOW_CLIENT = mlflow_client
    try:
        runs = MLFLOW_CLIENT.search_runs(
            experiment_ids = experiment_ids,
            filter_string = filter,
            run_view_type = type,
            max_results = max_amount
        )
        run_dict = {}
        for run in runs:
            run_dict[run.info.run_id] = {
                'e_id': run.info.experiment_id,
                'id': run.info.run_id,
                'name': run.info.run_name,
                'stage': run.info.lifecycle_stage,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'parameters': run.data.params,
                'metrics': run.data.metrics
            }
        return run_dict
    except Exception as e:
        logger.error('MLflow run getting error')
        logger.error(e)
        return None