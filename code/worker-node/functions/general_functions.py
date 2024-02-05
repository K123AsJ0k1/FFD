from flask import current_app

def get_debug_mode() -> list:
   return current_app.config['DEBUG']