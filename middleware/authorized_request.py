from functools import wraps
from fasthtml.common import *
from utils.jwt import verify_token

def authorized_request(f):
    @wraps(f)
    def decorated_function(session, *args, **kwargs):
        # First check if token exists in session
        token = session.get('AuthToken')
        
        # If not in session, try to get from authorization header
        if not token:
            return Redirect('/login')

        # Validate JWT token
        try:
            # Verify JWT token and get payload
            payload = verify_token(token)
            print(payload)
            if not payload:
                return Div(
                    "Access denied - Invalid token",
                    id="error",
                    style="color: red;"
                )
            
            print(payload)
            # Store user ID in session for later use
            session['uuid'] = payload['uuid']
            
            # Call the original function
            return f(session, *args, **kwargs)
            
        except Exception as e:
            print(f"Token validation error: {str(e)}")
            return Div(
                "Access denied - Token validation failed",
                id="error",
                style="color: red;"
            )
            
    return decorated_function