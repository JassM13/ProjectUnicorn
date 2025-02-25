from fasthtml.common import *
from views.components.sidebar import sidebar
from views.tradesview import trades_view
from profitpath_managers.trade_manager.trade_operations import TradeOperations
from models.trade import Trade
from datetime import datetime
from middleware.authorized_request import authorized_request
from middleware.device_restriction import restrict_small_devices

trade_operations = TradeOperations()

def register_trades_routes(rt):
    @rt("/trades")
    @authorized_request
    @restrict_small_devices()
    def get(session):
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
            # Create trade data dictionary
            trade_data = {
                'id': str(Trade().id),
                'symbol': form_data.get('symbol'),
                'entry_date': datetime.fromisoformat(form_data.get('entered_at')),
                'exit_date': datetime.fromisoformat(form_data.get('exited_at')),
                'trade_type': form_data.get('type').lower(),
                'entry_price': float(form_data.get('entry_price', 0)),
                'exit_price': float(form_data.get('exit_price', 0)),
                'position_size': float(form_data.get('size', 0))
            }
            
            if trade_operations.add_trade('default_user', trade_data):
                return Div(
                    f"Trade Added Successfully",
                    style="padding: 16px; background: #33cc33; color: white; border-radius: 8px;"
                )
            else:
                raise Exception("Failed to add trade")
        except Exception as e:
            return Div(
                f"Error adding trade: {str(e)}",
                style="padding: 16px; background: #ff4444; color: white; border-radius: 8px;"
            )
        
    @rt("/api/trades/{trade_id}")
    def delete(trade_id: str):
        try:
            if trade_operations.remove_trade('default_user', trade_id):
                return ""
            else:
                raise Exception("Failed to remove trade")
        except Exception as e:
            return Div(
                f"Error removing trade: {str(e)}",
                style="padding: 16px; background: #ff4444; color: white; border-radius: 8px;"
            )
    
    return rt