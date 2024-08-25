from functools import wraps
from flask import request, jsonify
import jwt

import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv('JWT_SECRET_KEY')


def token_required(role):
    def decorator(f):
        def decorated(*args, **kwargs):
            # Removed token and role checking logic
            return f(*args, **kwargs)
        return decorated
    return decorator
