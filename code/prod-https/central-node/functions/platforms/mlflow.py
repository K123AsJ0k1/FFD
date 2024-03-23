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

def start_experiment(
    experiment_name: str,
    experiemnt_tags: dict
) -> int:
    current_app.logger.info('Starting a experiment in MLflow')
    MLFLOW_CLIENT = current_app.mlflow_client 
    try:
        experiment_id = MLFLOW_CLIENT.create_experiment(
            name = experiment_name,
            tags = experiemnt_tags
        )
        current_app.logger.info('Starting succeeded')
        return experiment_id
    except Exception as e:
        current_app.logger.error('MLflow experiment starting error')
        current_app.logger.error(e)
        return None

def check_experiment(
    experiment_name: str
) -> dict:
    current_app.logger.info('Checking a experiment in MLflow')
    MLFLOW_CLIENT = current_app.mlflow_client 
    try:
        experiment_object = MLFLOW_CLIENT.get_experiment_by_name(
            name = experiment_name
        )
        current_app.logger.info('Checking succeeded')
        return experiment_object
    except Exception as e:
        current_app.logger.error('MLflow experiment checking error')
        current_app.logger.error(e)
        return None

def get_experiments(
    experiment_type: int,
    max_amount: int,
    filter: str
) -> dict:
    # Types are ACTIVE_ONLY = 1, DELETED_ONLY = 2 and ALL = 3
    current_app.logger.info('Getting experiments from MLflow')
    MLFLOW_CLIENT = current_app.mlflow_client 
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
        current_app.logger.info('Getting experiments succeeded')
        return experiment_dict
    except Exception as e:
        current_app.logger.error('MLflow experiment getting error')
        current_app.logger.error(e)
        return None

def start_run(
    experiment_id: str,
    tags: dict,
    name: str
) -> dict:
    current_app.logger.info('Starting a run in MLflow')
    MLFLOW_CLIENT = current_app.mlflow_client 
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
        current_app.logger.info('Starting succeeded')
        return run_dict
    except Exception as e:
        current_app.logger.error('MLflow run starting error')
        current_app.logger.error(e)
        return None

def check_run(
    run_id: str
) -> dict:
    current_app.logger.info('Checking a run in MLflow')
    MLFLOW_CLIENT = current_app.mlflow_client 
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
        current_app.logger.error('MLflow run checking error')
        current_app.logger.error(e)
        return None

def update_run(
    run_id: str,
    parameters: dict,
    metrics: dict
) -> bool:
    current_app.logger.info('Updating a run in MLflow')
    MLFLOW_CLIENT = current_app.mlflow_client 
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
        current_app.logger.info('Updating succeeded')
        return True
    except Exception as e:
        current_app.logger.error('MLflow run updating error')
        current_app.logger.error(e)
        return False
    
def end_run(
    run_id: str,
    status: str
) -> bool:
    # run status are FAILED = 4, FINISHED = 3, KILLED = 5, RUNNING = 1 and SCHEDULED = 2
    current_app.logger.info('Ending a run in MLflow')
    MLFLOW_CLIENT = current_app.mlflow_client
    try:
        MLFLOW_CLIENT.set_terminated(
            run_id = run_id,
            status = status
        )
        current_app.logger.info('Ending succeeded')
        return True
    except Exception as e:
        current_app.logger.error('MLflow run ending error')
        current_app.logger.error(e)
        return False

def get_runs(
    experiment_ids: list,
    filter: str,
    type: int,
    max_amount: int
) -> dict:
    current_app.logger.info('Getting runs from MLflow')
    MLFLOW_CLIENT = current_app.mlflow_client 
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
        current_app.logger.info('Getting succeeded')
        return run_dict
    except Exception as e:
        current_app.logger.error('MLflow run getting error')
        current_app.logger.error(e)
        return None