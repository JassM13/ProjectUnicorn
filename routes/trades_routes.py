from fasthtml.common import *
from views.components.sidebar import sidebar
from views.tradesview import trades_view
from servicesmanager.trade_service import TradeService
from models.trade import Trade
from datetime import datetime

trade_service = TradeService()

def register_trades_routes(rt):
    @rt("/trades")
    def get():
        return Div(
            sidebar(active="trades"),
            Div(trades_view(), style="margin-left: 100px;"),
            style="display:flex;"
        )
    
    @rt("/api/trades")
    async def post(request):
        try:
            # Extract form data
            form_data = await request.form()
            print(form_data)
            # Create trade object with proper type conversions
            trade = Trade(
                symbol=form_data.get('symbol'),
                entered_at=datetime.fromisoformat(form_data.get('entered_at')),
                exited_at=datetime.fromisoformat(form_data.get('exited_at')),
                instrument_type=form_data.get('instrument_type'),
                trade_day=datetime.fromisoformat(form_data.get('trade_day')),
                type=form_data.get('type'),
                entry_price=float(form_data.get('entry_price', 0)),
                exit_price=float(form_data.get('exit_price', 0)),
                size=float(form_data.get('size', 0))
            )
            trade_service.add_trade(trade)
            return Div(
                f"Trade Added Successfully",
                style="padding: 16px; background: #33cc33; color: white; border-radius: 8px;"
            )
        except Exception as e:
            return Div(
                f"Error adding trade: {str(e)}",
                style="padding: 16px; background: #ff4444; color: white; border-radius: 8px;"
            )
        
        
    @rt("/api/trades/{trade_id}")
    def delete(trade_id: str):
        try:
            trade_service.remove_trade(trade_id)
            return ""
        except Exception as e:
            return Div(
                f"Error removing trade: {str(e)}",
                style="padding: 16px; background: #ff4444; color: white; border-radius: 8px;"
            )
    
    return rt