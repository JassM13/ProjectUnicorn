from fasthtml.common import *
from views.dashboard_view.widgets.stats import stats_widget

def register_stats_route(rt):
    @rt("/api/widgets/stats")
    def get_stats_widget():
        return stats_widget()
    
    return rt