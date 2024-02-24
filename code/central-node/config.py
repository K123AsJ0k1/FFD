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
    # Needs to be refactored to have cycle and metrics thresholds
    CENTRAL_PARAMETERS = {
        'sample-pool': 500000,
        'train-eval-ratio': 0.5,
        'train-test-ratio': 0.8,
        'min-update-amount': 1,
        'max-cycles':2,
        'min-metric-sucess': 3,
        'metric-thresholds': {
            'true-positives': 2,
            'false-positives': 2,
            'true-negatives': 2,
            'false-negatives': 2,
            'recall': 0.05,
            'selectivity': 0.05,
            'precision': 0.05,
            'miss-rate': 0.05,
            'fall-out': 0.05,
            'balanced-accuracy': 0.05,
            'accuracy': 0.05
        }
    }

    WORKER_PARAMETERS = {
        'sample-pool': 100000,
        'train-test-ratio': 0.8
    }

class DevConfig(Config):
    DEBUG = True

class ProdConfig(Config):
    DEBUG = False