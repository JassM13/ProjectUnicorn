from fasthtml.common import *

def device_restriction():
    return Div(
        Div(
            H1("422", style="font-size: 8em; margin: 0; color: var(--primary-light); text-shadow: 2px 2px 10px rgba(246, 205, 112, 0.3); animation: float 6s ease-in-out infinite;"),
            H2("Device Not Supported", style="color: var(--text-primary); margin: 0; font-size: 2em;"),
            P("This magical experience requires a larger screen size...", style="color: var(--text-secondary); margin: 20px 0;"),
            P("Please visit us on a tablet or desktop device.", style="color: var(--text-secondary); margin: 10px 0 20px 0;"),
            style="text-align: center;"
        ),
        Style("""
            @keyframes float {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-20px); }
                100% { transform: translateY(0px); }
            }
        """),
        style="display: flex; justify-content: center; align-items: center; min-height: 100vh; background: #000; padding: 20px;"
    )