from fasthtml.common import *

def register_logout_routes(rt):
    @rt("/logout")
    def get(session):
        # Clear all session data for better security
        session.clear()
        
        return Redirect('/')
    
    return rt