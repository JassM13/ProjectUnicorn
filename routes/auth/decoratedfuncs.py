from functools import wraps
from fasthtml.common import *

def login_required(f):
    @wraps(f)
    def decorated_function(session, *args, **kwargs):
        if 'auth_token' not in session:
            return Redirect('/login')
        return f(session, *args, **kwargs)
    return decorated_function