from fasthtml.common import *
from monsterui.all import *

def index_view():
    return Div(
        # Theme toggle and styling
        Style("""
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .animate-fade-in {
                animation: fadeIn 0.8s ease-out forwards;
            }
        """),
        
        # Main container
        Div(
            # Hero Section
            Div(
                H2("Welcome to ProfitPath",
                   style="font-size: 3.5em; font-weight: 800; color: hsl(var(--primary)); margin-bottom: 20px;",
                   cls="animate-fade-in"),
                P("Your Journey to Financial Success Starts Here",
                  style="font-size: 1.5em; color: white; margin-bottom: 40px;",
                  cls="animate-fade-in"),
                
                # CTA Buttons
                Div(
                    A(Button("Get Started",
                            style="background: hsl(var(--primary)); color: hsl(var(--background)); \
                                   padding: 15px 40px; border-radius: 30px; font-weight: 600; \
                                   transition: all 0.3s ease;"),
                      href="/register",
                      style="text-decoration: none; margin-right: 20px;"),
                    A(Button("Login",
                            style="background: transparent; color: hsl(var(--primary)); \
                                   padding: 14px 38px; border-radius: 30px; font-weight: 600; \
                                   border: 2px solid hsl(var(--primary)); transition: all 0.3s ease;"),
                      href="/login",
                      style="text-decoration: none;"),
                    style="display: flex; justify-content: center; margin-bottom: 60px;",
                    cls="animate-fade-in"
                ),
                
                # Features Grid
                Div(
                    *[Div(
                        H3(title,
                           style="font-size: 1.5em; color: hsl(var(--primary)); margin-bottom: 15px;"),
                        P(description,
                          style="color: hsl(var(--secondary-foreground)); line-height: 1.6;"),
                        style="background: rgba(246, 205, 112, 0.1); padding: 30px; border-radius: 15px; \
                               text-align: center; transition: transform 0.3s ease;",
                        cls="animate-fade-in"
                    ) for title, description in [
                        ("Trade Analysis",
                         "Document and analyze your trades with detailed insights and performance metrics"),
                        ("Journal Sharing",
                         "Learn and grow by sharing your trading journal and experiences with fellow traders"),
                        ("Trade History",
                         "Keep comprehensive records of your trades with notes, screenshots, and outcomes")
                    ]],
                    style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); \
                           gap: 30px; margin-top: 40px;"
                )
            ),
            style="max-width: 1200px; margin: 0 auto; padding: 100px 20px 20px 20px;"
        ),
        style="min-height: 100vh; background: var(--background-dark); color: var(--text-primary); \
               font-family: system-ui, -apple-system, sans-serif;"
    )
