from flask import current_app
from typing import Dict

def get_debug_mode() -> list:
   return current_app.config['DEBUG']