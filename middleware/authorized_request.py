from functools import wraps
from fasthtml.common import *
from utils.jwt import verify_token
from database.firebase_manager import FirebaseManager

def authorized_request(f):
    @wraps(f)
    def decorated_function(session, *args, **kwargs):
        # First check if token exists in session
        token = session.get('auth_token')
        
        # If not in session, try to get from authorization header
        if not token:
            return Redirect('/login')

        
        # Verify JWT token and get payload
        payload = verify_token(token)
        if not payload:
            session['user_id'] = None
            session['auth_token'] = None
            return Redirect('/login')
            
        # Store user ID in session for later use
        session['user_id'] = payload['uuid']
            
        # Verify account exists in database
        firebase_manager = FirebaseManager.getInstance()
        user_ref = firebase_manager.db.collection('users').document(payload['uuid']).get()
        if not user_ref.exists:
            session['user_id'] = None
            session['auth_token'] = None
            return Redirect('/login')

        # Call the original function
        return f(session, *args, **kwargs)
            
            
    return decorated_function