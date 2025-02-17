from fasthtml.common import *
from views.components.sidebar import sidebar
from views.dashboardview import dashboard_view
from views.settings import settings_view
from routes.auth.jwt_auth import login_required

def register_dashboard_routes(rt):
    @rt("/dashboard")
    @login_required
    def get():
        return Div(
            sidebar(active="dashboard"),
            Div(dashboard_view(), style="margin-left: 100px;"),
            style="display:flex;"
        )

    @rt("/settings")
    @login_required
    def get():
        return Div(
            sidebar(active="settings"),
            Div(settings_view(), style="margin-left: 100px;"),
            style="display:flex;"
        )

    return rt