from fasthtml.common import *
from .models import User, validate_user
from views.auth.register import register_view

def register_register_routes(rt):
    @rt("/register")
    def get():
        return register_view()

    @rt("/auth/register")
    def post(user: User):
        errors = validate_user(user)
        if errors:
            return Div(
                Ul(*[Li(error) for error in errors]),
                id="result",
                style="color: red;"
            )
        # TODO: Implement actual user registration
        return Div(
            f"Account created successfully!",
            id="result",
            style="color: #f6cd70;"
        )

    return rt