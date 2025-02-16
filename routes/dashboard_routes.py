from fasthtml.common import *
from views.components.sidebar import sidebar
from views.dashboardview import dashboard_view
from views.settings import settings_view

def register_dashboard_routes(rt):
    @rt("/dashboard")
    def get_dashboard():
        return Div(
            sidebar(active="dashboard"),
            Div(dashboard_view(), style="margin-left: 100px;"),
            style="display:flex;"
        )

    @rt("/settings")
    def get_settings():
        return Div(
            sidebar(active="settings"),
            Div(settings_view(), style="margin-left: 100px;"),
            style="display:flex;"
        )

    return rt