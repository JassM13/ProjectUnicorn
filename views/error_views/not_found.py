from fasthtml.common import *

def not_found(req, exc):
    return Div(
        Div(
            H1("404", style="font-size: 8em; margin: 0; color: #f6cd70; text-shadow: 2px 2px 10px rgba(246, 205, 112, 0.3); animation: float 6s ease-in-out infinite;"),
            H2("Oops! Page Not Found", style="color: white; margin: 0; font-size: 2em;"),
            P("Looks like this unicorn wandered off to unknown territories...", style="color: #999; margin: 20px 0;"),
            A("Take Me Home", href="/", style="display: inline-block; background: #f6cd70; color: #000; text-decoration: none; padding: 12px 32px; border-radius: 8px; font-weight: bold; transition: transform 0.3s ease;"),
            style="text-align: center;"
        ),
        Style("""
            @keyframes float {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-20px); }
                100% { transform: translateY(0px); }
            }
            a:hover { transform: scale(1.05); }
        """),
        style="display: flex; justify-content: center; align-items: center; min-height: 100vh; background: #000; padding: 20px;"
    )