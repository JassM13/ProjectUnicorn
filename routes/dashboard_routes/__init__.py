from fasthtml.common import *
from views.main_view.mainview import mainview
from middleware.authorized_request import authorized_request
from middleware.device_restriction import restrict_small_devices
from routes.dashboard_routes.chart_routes import register_chart_routes

def register_dashboard_routes(rt):
    @rt("/dashboard")
    @authorized_request
    @restrict_small_devices()
    def get(session):
        return mainview(session, active="dashboard")

    @rt("/calendar")
    @authorized_request
    @restrict_small_devices()
    def get(session):
        return mainview(session, active="calendar")

    # Register all dashboard-related routes
    rt = register_chart_routes(rt)

    return rt