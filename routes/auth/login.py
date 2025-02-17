from fasthtml.common import *
from .models import User
from views.auth.login import login_view

def register_login_routes(rt):
    @rt("/login")
    def get():
        return login_view()

    @rt("/auth/login")
    def post(user: User):
        if not user.email or not user.password:
            return Div("Please fill in all fields", id="result", style="color: red;")
        return Div("Login successful!", id="result", style="color: #f6cd70;")

    return rt