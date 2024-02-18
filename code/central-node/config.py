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
        'scaled-columns': [
            'amount'
        ],
        'learning-rate': 0.001,
        'sample-rate': 0.01,
        'optimizer':'SGD',
        'epochs': 5
    }

    CENTRAL_PARAMETERS = {
        'sample-pool': 100000,
        'train-eval-ratio': 0.5,
        'train-test-ratio': 0.8,
        'min-update-amount': 1
    }

    WORKER_PARAMETERS = {
        'sample-pool': 10000,
        'train-test-ratio': 0.8
    }

class DevConfig(Config):
    DEBUG = True

class ProdConfig(Config):
    DEBUG = False