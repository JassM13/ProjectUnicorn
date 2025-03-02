from fasthtml.common import *
from views.error_views.device_restriction import device_restriction
from middleware.device_restriction import restrict_small_devices

def register_exception_routes(rt):
    @rt("/device-restricted")
    def get():
        return Div(
            Div(device_restriction()),
        )

    return rt