from dotenv import load_dotenv
from fasthtml.common import *
from fh_plotly import plotly_headers

from views.indexview import index_view
from views.error_views.not_found import not_found
from views.error_views.device_restriction import device_restriction

# Import route registrations
from routes.auth_routes import register_auth_routes
from routes.dashboard_routes import register_dashboard_routes
from routes.trades_routes import register_trades_routes
from routes.exceptions_routes import register_exception_routes
from routes.profiles_routes import register_profile_routes
from routes.account_routes import register_account_routes


load_dotenv()
custom_theme_css = Link(rel="stylesheet", href="static/css/theme.css", type="text/css")

exception_handlers = {404: not_found,
                      422: device_restriction}

app, rt = fast_app(live=bool(os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"),
                  secret_key=os.getenv("SESSION_SECRET_KEY"),
                  hdrs=(
                    Script(defer=True, src="static/js/alpine.min.js"),
                    plotly_headers, picolink,
                    Link(rel="stylesheet", href="/static/css/theme.css", type="text/css"),
                    Link(rel="icon", type="image/x-icon", href="/assets/favicon.svg"),
                    SortableJS('.sortable')),
                    exception_handlers=exception_handlers
               )

app.title="ProfitPath"

# Register all routes
rt = register_auth_routes(rt)
rt = register_dashboard_routes(rt)
rt = register_trades_routes(rt)
rt = register_exception_routes(rt)
rt = register_profile_routes(rt)
rt = register_account_routes(rt)

# Default route
@rt("/")
def get_home(session):
    return index_view(session)

# Serve the app
serve()