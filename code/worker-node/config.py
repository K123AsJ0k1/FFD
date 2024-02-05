class Config:
    DEBUG = None
    CENTRAL_ADDRESS = 'http://0.0.0.0:7600'
    
class DevConfig(Config):
    DEBUG = True

class ProdConfig(Config):
    DEBUG = False