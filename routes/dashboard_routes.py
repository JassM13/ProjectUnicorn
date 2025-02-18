from fasthtml.common import *
from views.components.sidebar import sidebar
from views.dashboardview import dashboard_view
from views.settings import settings_view
from routes.auth.decoratedfuncs import login_required

def register_dashboard_routes(rt):
    @rt("/dashboard")
    @login_required
    def get(session):
        if 'auth_token' not in session:
            return Redirect('/login')
        return Div(
            sidebar(active="dashboard"),
            Div(dashboard_view(), style="margin-left: 100px;"),
            style="display:flex;"
        )

    @rt("/settings")
    def get(session):
        return Div(
            sidebar(active="settings"),
            Div(settings_view(), style="margin-left: 100px;"),
            style="display:flex;"
        )

    return rt