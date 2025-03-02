from .login import register_login_routes
from .register import register_register_routes

def register_auth_routes(rt):
    rt = register_login_routes(rt)
    rt = register_register_routes(rt)
    return rt