from functools import wraps
from fasthtml.common import *
from utils.jwt import verify_token
from database.firebase_manager import FirebaseManager
import asyncio
import json

def authorized_request(f):
    @wraps(f)
    async def async_decorated_function(session, *args, **kwargs):
        # Check if this is an API route that should return JSON
        is_api_route = 'request' in kwargs and '/api/' in str(kwargs['request'].url)
        
        # First check if token exists in session
        token = session.get('auth_token')
        
        # If not in session, handle the unauthorized access
        if not token:
            if is_api_route:
                error_json = json.dumps({"error": "Authentication required"})
                return Response(error_json, content_type="application/json")
            return Redirect('/login')
        
        # Verify JWT token and get payload
        payload = verify_token(token)
        if not payload:
            session['user_id'] = None
            session['auth_token'] = None
            if is_api_route:
                error_json = json.dumps({"error": "Invalid or expired authentication token"})
                return Response(error_json, content_type="application/json")
            return Redirect('/login')
            
        # Store user ID in session for later use
        session['user_id'] = payload['uuid']
            
        # Verify account exists in database
        firebase_manager = FirebaseManager.getInstance()
        user_ref = firebase_manager.db.collection('users').document(payload['uuid']).get()
        if not user_ref.exists:
            session['user_id'] = None
            session['auth_token'] = None
            if is_api_route:
                error_json = json.dumps({"error": "User account not found"})
                return Response(error_json, content_type="application/json")
            return Redirect('/login')

        # Call the original function - handle both async and sync functions
        if asyncio.iscoroutinefunction(f):
            return await f(session, *args, **kwargs)
        else:
            return f(session, *args, **kwargs)
    
    @wraps(f)
    def sync_decorated_function(session, *args, **kwargs):
        # Check if this is an API route that should return JSON
        is_api_route = 'request' in kwargs and '/api/' in str(kwargs['request'].url)
        
        # First check if token exists in session
        token = session.get('auth_token')
        
        # If not in session, handle the unauthorized access
        if not token:
            if is_api_route:
                error_json = json.dumps({"error": "Authentication required"})
                return Response(error_json, content_type="application/json")
            return Redirect('/login')
        
        # Verify JWT token and get payload
        payload = verify_token(token)
        if not payload:
            session['user_id'] = None
            session['auth_token'] = None
            if is_api_route:
                error_json = json.dumps({"error": "Invalid or expired authentication token"})
                return Response(error_json, content_type="application/json")
            return Redirect('/login')
            
        # Store user ID in session for later use
        session['user_id'] = payload['uuid']
            
        # Verify account exists in database
        firebase_manager = FirebaseManager.getInstance()
        user_ref = firebase_manager.db.collection('users').document(payload['uuid']).get()
        if not user_ref.exists:
            session['user_id'] = None
            session['auth_token'] = None
            if is_api_route:
                error_json = json.dumps({"error": "User account not found"})
                return Response(error_json, content_type="application/json")
            return Redirect('/login')

        # Call the original function
        return f(session, *args, **kwargs)
    
    # Return the appropriate wrapper based on whether the wrapped function is async or not
    if asyncio.iscoroutinefunction(f):
        return async_decorated_function
    else:
        return sync_decorated_function