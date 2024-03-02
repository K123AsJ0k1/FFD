class Config:
    DEBUG = None
    
class DevConfig(Config):
    DEBUG = True

class ProdConfig(Config):
    DEBUG = False