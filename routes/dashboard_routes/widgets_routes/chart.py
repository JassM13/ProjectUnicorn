from fasthtml.common import *
from views.dashboard_view.widgets.chart import chart_widget

def register_chart_route(rt):
    @rt("/api/widgets/chart")
    def get_chart_widget():
        return chart_widget()
    
    return rt