from .profile_route import register_profile_routes

def register_auth_routes(rt):
    rt = register_profile_routes(rt)
    return rt