from fasthtml.common import *
from views.components.sidebar import sidebar
from views.dashboardview import dashboard_view
from views.calendarview import calendar_view
from views.errors.device_restriction import device_restriction
from middleware.device_restriction import restrict_small_devices

def register_exception_routes(rt):
    @rt("/device-restricted")
    def get():
        return Div(
            Div(device_restriction()),
        )

    return rt