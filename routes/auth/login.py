from fasthtml.common import *
from dataclasses import dataclass
from .storage import UserStorage
from utils.jwt import generate_token
from views.auth.login import login_view
from models.user import User

def register_login_routes(rt):
    storage = UserStorage()

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
            if not storage.verify_password(user.email, user.password, is_email=True):
                return Div("Invalid email or password", id="result", style="color: red;")
            identifier = user.email
        else:
            if not storage.verify_password(user.username, user.password):
                return Div("Invalid username or password", id="result", style="color: red;")
            identifier = user.username
        token = generate_token(user.user_id)
        session['AuthToken'] = token
        return Redirect('/dashboard')

    return rt