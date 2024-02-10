class Config:
    DEBUG = None

    GLOBAL_SEED = 42
    GLOBAL_LEARNING_RATE = 0.001
    GLOBAL_SAMPLE_RATE = 0.01
    GLOBAL_MODEL_OPTIMIZER = 'SGD'
    GLOBAL_TRAINING_EPOCHS = 5

    SCALED_COLUMNS = [
        'amount'
    ]
    USED_COLUMNS = [
        'amount',
        'type_CASH_IN',
        'type_CASH_OUT',
        'type_DEBIT',
        'type_PAYMENT',
        'type_TRANSFER',
        'isFraud'
    ]
    TARGET_COLUMN = 'isFraud'
    INPUT_SIZE = 6

    CENTRAL_SAMPLE_POOL = 10000
    CENTRAL_TRAIN_EVALUATION_RATIO = 0.5
    CENTRAL_TRAIN_TEST_RATIO = 0.8
    
    WORKER_SAMPLE_POOL = 100
    WORKER_TRAIN_TEST_RATIO = 0.8

    AVAILABLE_WORKERS = []
    
class DevConfig(Config):
    DEBUG = True

class ProdConfig(Config):
    DEBUG = False