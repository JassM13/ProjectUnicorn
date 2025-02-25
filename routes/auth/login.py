from fasthtml.common import *
from dataclasses import dataclass
from profitpath_managers.user_manager.user_authentication_service import UserAuthenticationService
from utils.jwt import generate_token
from views.auth.login import login_view
from models.user import User

def register_login_routes(rt):
    auth_service = UserAuthenticationService()

    @rt("/login")
    def get(session):
        if 'AuthToken' in session:
            return Redirect('/dashboard')
        return login_view()

    @rt("/auth/login")
    def post(session, user: User):
        errors = User.validate(user)
        if errors:
            return Div(errors, id="result", style="color: red;")
        
        print(user.identifier)
        # Try to verify password with either username or email
        if '@' in user.identifier:
            if not auth_service.verify_password(user.identifier, user.password, is_email=True):
                return Div("Invalid email or password", id="result", style="color: red;")
            identifier = user.email
        else:
            if not auth_service.verify_password(user.identifier, user.password):
                return Div("Invalid username or password", id="result", style="color: red;")
            identifier = user.username
        token = generate_token(user.user_id)
        session['AuthToken'] = token
        return Redirect('/dashboard')

    return rt