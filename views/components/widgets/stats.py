from fasthtml.common import *
from servicesmanager.trade_service import TradeService

def stats_widget(user_id=None):
    trade_service = TradeService()
    stats = trade_service.calculate_stats(user_id) if user_id else {
        'win_rate': 68.5,
        'avg_position_size': 5420,
        'risk_reward': 2.5,
        'total_trades': 300
    }

    return Card(
        Div(
            Div(
                Div(
                    P("Win Rate", style="margin: 0; color: #999; font-size: clamp(0.8rem, 1.5vw, 0.9rem); font-weight: 500; margin-bottom: 4px;"),
                    P(f"{stats['win_rate']}%", style="margin: 0; font-size: clamp(1.2rem, 2vw, 1.5rem); color: #4CAF50; font-weight: 600;"),
                    style="text-align: center; background-color: rgba(8, 8, 8, 1);; padding: clamp(8px, 2vw, 16px); border-radius: 8px; transition: transform 0.2s; cursor: pointer; &:hover { transform: translateY(-2px); }; min-width: 0;"
                ),
                Div(
                    P("Avg Position Size", style="margin: 0; color: #999; font-size: clamp(0.8rem, 1.5vw, 0.9rem); font-weight: 500; margin-bottom: 4px;"),
                    P(f"${stats['avg_position_size']:,.0f}", style="margin: 0; font-size: clamp(1.2rem, 2vw, 1.5rem); color: #f6cd70; font-weight: 600;"),
                    style="text-align: center; background-color: rgba(8, 8, 8, 1);; padding: clamp(8px, 2vw, 16px); border-radius: 8px; transition: transform 0.2s; cursor: pointer; &:hover { transform: translateY(-2px); }; min-width: 0;"
                ),
                Div(
                    P("Risk/Reward", style="margin: 0; color: #999; font-size: clamp(0.8rem, 1.5vw, 0.9rem); font-weight: 500; margin-bottom: 4px;"),
                    P(f"1:{stats['risk_reward']}", style="margin: 0; font-size: clamp(1.2rem, 2vw, 1.5rem); color: #f6cd70; font-weight: 600;"),
                    style="text-align: center; background-color: rgba(8, 8, 8, 1);; padding: clamp(8px, 2vw, 16px); border-radius: 8px; transition: transform 0.2s; cursor: pointer; &:hover { transform: translateY(-2px); }; min-width: 0;"
                ),
                Div(
                    P("Total Trades", style="margin: 0; color: #999; font-size: clamp(0.8rem, 1.5vw, 0.9rem); font-weight: 500; margin-bottom: 4px;"),
                    P(f"{stats['total_trades']}", style="margin: 0; font-size: clamp(1.2rem, 2vw, 1.5rem); color: #f6cd70; font-weight: 600;"),
                    style="text-align: center; background-color: rgba(8, 8, 8, 1);; padding: clamp(8px, 2vw, 16px); border-radius: 8px; transition: transform 0.2s; cursor: pointer; &:hover { transform: translateY(-2px); }; min-width: 0;"
                ),
                style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: clamp(12px, 2vw, 20px); margin-top: clamp(16px, 3vw, 24px); width: 100%;"
            ),
            style="background: linear-gradient(315deg, rgba(246,205,112, 0.01) 0%, rgba(246,205,112, .2) 100%); padding: clamp(16px, 3vw, 24px); height: 100%; display: flex; flex-direction: column;"
        ),
        style="background-color: #000; height: 100%; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);"
    )