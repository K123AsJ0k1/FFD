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
        'learning-rate': 0.005,
        'sample-rate': 0.10,
        'optimizer':'SGD',
        'epochs': 5
    }
    
    CENTRAL_PARAMETERS = {
        'sample-pool': 50000,
        'train-eval-ratio': 0.5,
        'train-test-ratio': 0.8,
        'min-update-amount': 1,
        'max-cycles':2,
        'min-metric-success': 6,
        'metric-thresholds': {
            'true-positives': 50,
            'false-positives': 100,
            'true-negatives': 1000, 
            'false-negatives': 100,
            'recall': 0.40,
            'selectivity': 0.99,
            'precision': 0.80,
            'miss-rate': 0.05,
            'fall-out': 0.05,
            'balanced-accuracy': 0.85,
            'accuracy': 0.99
        },
        'metric-conditions': {
            'true-positives': '>=',
            'false-positives': '<=',
            'true-negatives': '>=', 
            'false-negatives': '<=',
            'recall': '>=',
            'selectivity': '>=',
            'precision': '>=',
            'miss-rate': '<=',
            'fall-out': '<=',
            'balanced-accuracy': '>=',
            'accuracy': '>='
        }
    }

    WORKER_PARAMETERS = {
        'sample-pool': 50000,
        'train-test-ratio': 0.8
    }

class DevConfig(Config):
    DEBUG = True

class ProdConfig(Config):
    DEBUG = False