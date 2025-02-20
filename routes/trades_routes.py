from fasthtml.common import *
from views.components.sidebar import sidebar
from views.tradesview import trades_view
import traceback
import asyncio

def register_trades_routes(rt):
    @rt("/trades")
    def get():
        return Div(
            sidebar(active="trades"),
            Div(trades_view(), style="margin-left: 100px;"),
            style="display:flex;"
        )
    
    return rt