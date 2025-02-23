from fasthtml.common import *
from views.indexview import index_view
from views.errors.not_found import not_found
from views.errors.device_restriction import device_restriction

# Import route registrations
from routes.auth import register_auth_routes
from routes.dashboard_routes import register_dashboard_routes
from routes.trades_routes import register_trades_routes
from routes.exceptions_routes import register_exception_routes

exception_handlers = {404: not_found,
                      422: device_restriction}

app, rt = fast_app(live=True,
                  secret_key='unicorn-project-secret-key-2024',
                  hdrs=(picolink,
                    Style(""":root {--pico-spacing: 0rem;} @media only screen and (prefers-color-scheme:dark){:root:not([data-theme]){--pico-background-color: rgba(246,205,112, 1);"""),
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