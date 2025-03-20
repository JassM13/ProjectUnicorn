from fasthtml.common import *
from views.dashboard_view.widgets.progress import progress_widget

def register_progress_route(rt):
    @rt("/api/widgets/progress")
    def get_progress_widget():
        return progress_widget()
    
    return rt