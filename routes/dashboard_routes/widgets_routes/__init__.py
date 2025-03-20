from fasthtml.common import *
from .total_profit import register_total_profit_route
from .progress import register_progress_route
from .chart import register_chart_route
from .stats import register_stats_route

def register_widget_routes(rt):
    rt = register_total_profit_route(rt)
    rt = register_progress_route(rt)
    rt = register_chart_route(rt)
    rt = register_stats_route(rt)
    return rt