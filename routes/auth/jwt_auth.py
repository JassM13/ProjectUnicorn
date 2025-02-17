import jwt
from datetime import datetime, timedelta
from functools import wraps
from fasthtml.common import *

# Secret key for JWT encoding/decoding - in production, this should be in environment variables
JWT_SECRET = 'unicorn-project-secret-key-2024'
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_DELTA = timedelta(days=1)

def generate_token(username: str) -> str:
    """Generate a new JWT token for a user"""
    payload = {
        'username': username,
        'exp': datetime.utcnow() + JWT_EXPIRATION_DELTA,
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str) -> dict:
    """Verify a JWT token and return the payload"""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def get_current_user() -> str:
    """Get the current user from the session cookie"""
    token = cookie('auth_token')
    if not token:
        return None
    payload = verify_token(token)
    if not payload:
        # Clear invalid token
        cookie('auth_token', '', expires=0)
        return None
    return payload['username']

def login_required(route_func):
    """Decorator to protect routes that require authentication"""
    @wraps(route_func)
    def wrapper(*args, **kwargs):
        current_user = get_current_user()
        if not current_user:
            # Redirect to login page if user is not authenticated
            return Redirect('/login')
        return route_func(*args, **kwargs)
    return wrapper