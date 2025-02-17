from fasthtml.common import *
from dataclasses import dataclass
from .storage import UserStorage
from .jwt_auth import generate_token
from views.auth.login import login_view
from models.user import User

def register_login_routes(rt):
    storage = UserStorage()

    @rt("/login")
    def get():
        return login_view()

    @rt("/auth/login")
    def post(user: User):
        is_valid, error = User.validate(user)
        if not is_valid:
            return Div(error, id="result", style="color: red;")
        
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
        
        # Generate JWT token and set it as a cookie
        print('made it here')
        token = generate_token(identifier)
        print(token)
        cookie('auth_token', token, httponly=False, secure=False)  # 24 hours
        
        return Redirect('/dashboard')

    return rt