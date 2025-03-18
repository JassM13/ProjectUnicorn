from fasthtml.common import *
from views.dashboard_view.widgets.chart import generate_line_chart
from fh_plotly import plotly2fasthtml
from middleware.authorized_request import authorized_request

def register_chart_routes(rt):
    @authorized_request
    @rt("/api/chart/refresh")
    def get(session):
        return plotly2fasthtml(generate_line_chart())
    
    return rt