from fasthtml.common import *
from dataclasses import dataclass
from profitpath_managers.user_manager.user_authentication_service import UserAuthenticationService
from views.auth.register import register_view
from models.user import User

def register_register_routes(rt):
    auth_service = UserAuthenticationService()

    @rt("/register")
    def get(session):
        if 'AuthToken' in session:
            return Redirect('/dashboard')
        return register_view()

    @rt("/auth/register")
    def post(user: User):
        errors = User.validate(user)
        print(f"errors: {errors}")
        if errors:
            return Div(errors, id="result", style="color: red;")
        
        if not auth_service.create_user(user):
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