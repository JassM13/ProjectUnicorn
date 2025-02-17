import jwt
from datetime import datetime, timedelta

# Secret key for JWT encoding/decoding - in production, this should be in environment variables
JWT_SECRET = 'unicorn-project-secret-key-2024'
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_DELTA = timedelta(days=1)

def generate_token(user_id: str) -> str:
    """Generate a new JWT token for a user"""
    payload = {
        'user_id': user_id,
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