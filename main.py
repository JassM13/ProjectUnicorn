from dotenv import load_dotenv
from fasthtml.common import *
from views.indexview import index_view
from views.error_views.not_found import not_found
from views.error_views.device_restriction import device_restriction

# Import route registrations
from routes.auth_routes import register_auth_routes
from routes.dashboard_routes import register_dashboard_routes
from routes.trades_routes import register_trades_routes
from routes.exceptions_routes import register_exception_routes


load_dotenv()

exception_handlers = {404: not_found,
                      422: device_restriction}

app, rt = fast_app(live=bool(os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"),
                  secret_key=os.getenv("SESSION_SECRET_KEY"),
                  hdrs=(picolink,
                    Style(""":root {
                                --pico-spacing: 0rem;
                                --primary-color: rgb(246, 205, 112);
                                --primary-light: rgb(246, 205, 112);
                                --primary-dark: rgba(246, 205, 112, 0.6);
                                --success-color: #33cc33;
                                --error-color: #ff4444;
                                --text-primary: #333333;
                                --text-secondary: #666666;
                                --background-light: #ffffff;
                                --background-dark: #000000;
                              }

                              @media only screen and (prefers-color-scheme: dark) {
                                :root:not([data-theme]) {
                                  --pico-background-color: var(--primary-color);
                                  --text-primary: #333333;
                                  --text-secondary: #666666;
                                  --background-color: var(--background-dark);
                                }
                              }

                              @media only screen and (prefers-color-scheme: light) {
                                :root:not([data-theme]) {
                                  --pico-background-color: var(--background-light);
                                  --text-primary: #333333;
                                  --text-secondary: #666666;
                                  --background-color: var(--background-light);
                                }
                              }"""),
                    SortableJS('.sortable')),
                    exception_handlers=exception_handlers
               )

# Register all routes
rt = register_auth_routes(rt)
rt = register_dashboard_routes(rt)
rt = register_trades_routes(rt)
rt = register_exception_routes(rt)

# Default route
@rt("/")
def get_home():
    return index_view()

# Serve the app
serve()