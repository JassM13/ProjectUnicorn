from fasthtml.common import *
from views.main_view.mainview import mainview
from middleware.authorized_request import authorized_request
from .create import register_create_profile_routes
from .delete import register_delete_profile_routes
from .get import register_get_profile_routes

def register_profile_routes(rt):
    @rt("/profiles")
    @authorized_request
    def get(session):
        return mainview(session, active="profiles")
    
    # Register all profile-related routes
    rt = register_create_profile_routes(rt)
    rt = register_delete_profile_routes(rt)
    rt = register_get_profile_routes(rt)
    
    return rt