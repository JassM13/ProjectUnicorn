from fasthtml.common import *

def login_view():
    return Div(
        H1("Login", cls="hero-text", style="font-size: 3em; margin-bottom: 24px; font-weight: 800; letter-spacing: -1px;"),
        P("Welcome back to Project Unicorn", style="color: white; font-size: 1.2em; margin-bottom: 40px;"),
        Form(
            Div(
                Input(
                    type="text",
                    name="identifier",
                    placeholder="Email/Username",
                    style="width: 100%; padding: 12px; margin-bottom: 16px; border-radius: 8px; \
                           background: #222; border: 1px solid #333; color: white;"
                ),
                Input(
                    type="password",
                    name="password",
                    placeholder="Password",
                    style="width: 100%; padding: 12px; margin-bottom: 16px; border-radius: 8px; \
                           background: #222; border: 1px solid #333; color: white;"
                ),
                Button(
                    "Login",
                    type="submit",
                    style="width: 100%; padding: 12px; background: #f6cd70; color: black; \
                           border: none; border-radius: 8px; font-weight: bold; cursor: pointer; \
                           transition: all 0.3s ease;"
                ),
                style="width: 100%; max-width: 400px;"
            ),
            Div(
                P(
                    "Don't have an account? ",
                    A("Sign up here", href="/register", style="color: #f6cd70; text-decoration: none;"),
                    style="margin-top: 20px; color: white;"
                ),
                id="result"
            ),
            hx_post="/auth/login",
            hx_target="#result",
            style="display: flex; flex-direction: column; align-items: center; width: 100%;"
        ),
        style="""padding: 20px; background-color: #000; color: white; height: 95vh;
                border-radius: 16px; display: flex; flex-direction: column;
                justify-content: center; align-items: center; position: absolute;
                right: 20px; top: 20px; left: 20px; bottom: 20px;"""
    )