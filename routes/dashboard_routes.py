from fasthtml.common import *
from views.components.sidebar import sidebar
from views.dashboardview import dashboard_view
from views.calendarview import calendar_view
from middleware.authorized_request import authorized_request
from middleware.device_restriction import restrict_small_devices

def register_dashboard_routes(rt):
    @rt("/dashboard")
    @authorized_request
    @restrict_small_devices()
    def get(session):
        return Div(
            sidebar(active="dashboard"),
            Div(dashboard_view(), style="margin-left: 100px;"),
            style="display:flex;"
        )

    @rt("/calendar")
    @authorized_request
    @restrict_small_devices()
    def get(session):
        return Div(
            sidebar(active="calendar"),
            Div(calendar_view(), style="margin-left: 100px;"),
            style="display:flex;"
        )

    return rt