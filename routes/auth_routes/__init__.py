from .login import register_login_routes
from .register import register_register_routes
from .logout import register_logout_routes

def register_auth_routes(rt):
    rt = register_login_routes(rt)
    rt = register_register_routes(rt)
    rt = register_logout_routes(rt)
    return rt