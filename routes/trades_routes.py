from fasthtml.common import *
from views.main_view.mainview import mainview
from models.trade import Trade, TradeGroup
from datetime import datetime
from middleware.authorized_request import authorized_request
from middleware.device_restriction import restrict_small_devices


def register_trades_routes(rt):
    @rt("/trades")
    @authorized_request
    @restrict_small_devices()
    def get(session):
        return mainview(active="trades")
    
    # Temporarily disabled /api/trades endpoint
    @rt("/api/trades")
    async def post(request, session):
        return Div(
            "Trade submission is temporarily disabled",
            style="padding: 16px; background: #ff9933; color: white; border-radius: 8px;"
        )
    
    """    
    # Original implementation preserved for reference
    @rt("/api/trades")
    async def post(request, session):
        try:
            # Get user's default sub-account
            user_id = session.get('user_id')
            
            # Extract form data
            form_data = await request.form()
            entered_at = datetime.fromisoformat(form_data.get('entered_at'))
            exited_at = datetime.fromisoformat(form_data.get('exited_at'))
            
            # Create new trade instance with all required parameters
            trade = Trade(
                symbol=form_data.get('symbol'),
                entered_at=entered_at,
                exited_at=exited_at,
                instrument_type=form_data.get('instrument_type', 'stock'),
                trade_day=entered_at.date(),
                type=form_data.get('type').lower(),
                entry_price=float(form_data.get('entry_price', 0)),
                exit_price=float(form_data.get('exit_price', 0)),
                size=float(form_data.get('size', 0))
            )
            
            # Create a trade group for the sub-account if it doesn't exist
            trade_group = TradeGroup()
            trade_group.trades.append(trade)
            
            # Create trade data dictionary for database operation
            trade_data = {
                'id': str(trade.id),
                'symbol': trade.symbol,
                'entry_date': trade.entered_at,
                'exit_date': trade.exited_at,
                'trade_type': trade.type,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'position_size': trade.size,
                #'sub_account_id': str(default_account.id),
                'trade_group_id': str(trade_group.id)
            }
            
            if trade_operations.add_trade(user_id, trade_data['sub_account_id'], trade_data):
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
    """

    
    return rt