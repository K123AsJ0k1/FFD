class Config:
    DEBUG = None

    GLOBAL_SEED = 42
    GLOBAL_LEARNING_RATE = 0.001
    GLOBAL_SAMPLE_RATE = 0.01
    GLOBAL_EPOCHS = 5

    SCALED_COLUMNS = [
        'amount'
    ]
    INPUT_COLUMNS = [
        'amount',
        'type_CASH_IN',
        'type_CASH_OUT',
        'type_DEBIT',
        'type_PAYMENT',
        'type_TRANSFER',
        'isFraud'
    ]
    OUTPUT_COLUMN = 'isFraud'

    CENTRAL_SAMPLE_POOL = 4000000
    CENTRAL_TRAIN_EVALUATION_RATIO = 0.5
    CENTRAL_TRAIN_TEST_RATIO = 0.8
    
    WORKER_SAMPLE_POOL = 1000000
    WORKER_TRAIN_TEST_RATIO = 0.8
    
    
class DevConfig(Config):
    DEBUG = True

class ProdConfig(Config):
    DEBUG = False