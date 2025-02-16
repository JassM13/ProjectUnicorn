from fasthtml.common import *
from dataclasses import dataclass

@dataclass
class LoginForm:
    username: str
    password: str


def login_view():
    return Div(
        Div(
            H1("Login", style="color: #f6cd70; margin-bottom: 24px; text-align: center;"),
            Form(
                Div(
                    Label("Username", For="username", style="color: #f6cd70;"),
                    Input(type="text", id="username", name="username", required=True,
                          style="width: 100%; padding: 8px; margin: 8px 0; background: #333; "
                                "border: 1px solid #f6cd70; border-radius: 4px; color: white;"),
                    style="margin-bottom: 16px;"
                ),
                Div(
                    Label("Password", For="password", style="color: #f6cd70;"),
                    Input(type="password", id="password", name="password", required=True,
                          style="width: 100%; padding: 8px; margin: 8px 0; background: #333; "
                                "border: 1px solid #f6cd70; border-radius: 4px; color: white;"),
                    style="margin-bottom: 24px;"
                ),
                Button("Login", type="submit",
                       style="width: 100%; padding: 12px; background: #f6cd70; color: #000; "
                             "border: none; border-radius: 4px; cursor: pointer; font-weight: bold;"),
                P(
                    "Don't have an account? ",
                    A("Sign up here", href="/signup", style="color: #f6cd70; text-decoration: none;"),
                    style="text-align: center; margin-top: 16px; color: white;"
                ),
                action="/login",
                method="post",
                style="width: 100%;"
            ),
            style="background: #222; padding: 32px; border-radius: 8px; width: 100%; max-width: 400px;"
        ),
        style="display: flex; justify-content: center; align-items: center; min-height: 100vh; "
              "background: #000; padding: 20px;"
    )