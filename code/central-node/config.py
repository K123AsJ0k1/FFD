class Config:
    DEBUG = None

    GLOBAL_PARAMETERS = {
        'seed': 42,
        'used-columns': [
            'amount',
            'type_CASH_IN',
            'type_CASH_OUT',
            'type_DEBIT',
            'type_PAYMENT',
            'type_TRANSFER',
            'isFraud'
        ],
        'input-size': 6,
        'target-column': 'isFraud',
        'scaled_columns': [
            'amount'
        ],
        'learning-rate': 0.001,
        'sample-rate': 0.01,
        'optimizer':'SGD',
        'epochs': 5
    }

    CENTRAL_PARAMETERS = {
        'sample-pool': 10000,
        'train-eval-ratio': 0.5,
        'train-test-ratio': 0.8
    }

    WORKER_PARAMETERS = {
        'sample-pool': 100,
        'train-test-ratio': 0.8,
        'available': []
    }

    #GLOBAL_SEED = 42
    #GLOBAL_USED_COLUMNS = [
    #    'amount',
    #    'type_CASH_IN',
    #    'type_CASH_OUT',
    ##    'type_DEBIT',
    #    'type_PAYMENT',
    #    'type_TRANSFER',
    #    'isFraud'
    #]
    #GLOBAL_INPUT_SIZE = 6
    #GLOBAL_TARGET_COLUMN = 'isFraud'
    #GLOBAL_SCALED_COLUMNS = [
    #    'amount'
    #]
    
    #GLOBAL_LEARNING_RATE = 0.001
    #GLOBAL_SAMPLE_RATE = 0.01
    #GLOBAL_MODEL_OPTIMIZER = 'SGD'
    #GLOBAL_TRAINING_EPOCHS = 5

    #CENTRAL_SAMPLE_POOL = 10000
    #CENTRAL_TRAIN_EVALUATION_RATIO = 0.5
    #CENTRAL_TRAIN_TEST_RATIO = 0.8
    
    #WORKER_SAMPLE_POOL = 100
    #WORKER_TRAIN_TEST_RATIO = 0.8

    #AVAILABLE_WORKERS = []
    
class DevConfig(Config):
    DEBUG = True

class ProdConfig(Config):
    DEBUG = False