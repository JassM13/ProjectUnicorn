from fasthtml.common import *
from dataclasses import dataclass
from .storage import UserStorage
from views.auth.register import register_view
from models.user import User

def register_register_routes(rt):
    storage = UserStorage()

    @rt("/register")
    def get(session):
        if 'auth_token' in session:
            return Redirect('/dashboard')
        return register_view()

    @rt("/auth/register")
    def post(user: User):
        errors = User.validate(user)
        print(errors)
        if errors:
            return Div(errors, id="result", style="color: red;")
        
        if not storage.create_user(user):
            return Div(
                "Username or email already exists",
                id="result",
                style="color: red;"
            )

        return Div(
            f"Account created successfully!",
            id="result",
            style="color: #f6cd70;"
        )

    return rt