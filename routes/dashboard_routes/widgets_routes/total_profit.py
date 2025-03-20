from fasthtml.common import *
from views.dashboard_view.widgets.total_profit import total_profit_widget

def register_total_profit_route(rt):
    @rt("/api/widgets/profit")
    def get_profit_widget():
        return total_profit_widget()
    
    return rt