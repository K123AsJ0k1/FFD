from app import create_app
import warnings
import os

warnings.filterwarnings("ignore")
# The application can be run with the command 'python3 run.py'
if __name__ == "__main__":
    # This application uses the app factory method
    # to make code easier to understand and refactor
    
    os.environ['CENTRAL_ADDRESS'] = '127.0.0.1'
    os.environ['CENTRAL_PORT'] = '7500'

    os.environ['WORKER_PORT'] = '7501'
    os.environ['WORKER_ADDRESS'] = '127.0.0.1'

    os.environ['WORKER_SYSTEM_MONITOR'] = '0'

    os.environ['MINIO_ENDPOINT'] = '127.0.0.1:9000'
    os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'

    os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:5000'
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'

    app = create_app()
    app.run(
        host = '0.0.0.0', 
        port = 7501
    )