from fasthtml.common import *

def stats_widget(user_id=None):
    stats = {
        'win_rate': 68.5,
        'avg_position_size': 2,
        'risk_reward': 2.5,
        'best_day': 3300,
        'worst_day': -1200,
        'total_trades': 300
    }

    return Card(
        Div(
            H2("Stats", style="margin: 0; color: rgba(246,205,112, 0.6); font-size: clamp(1.2rem, 2vw, 1.5rem); flex: 0 0 auto;"),
            Div(
                Div(
                    P("Best Day", style="margin: 0; color: #999; font-size: clamp(0.8rem, 1.5vw, 0.9rem); font-weight: 500; margin-bottom: 4px;"),
                    P(f"{stats['best_day']}", style="margin: 0; font-size: clamp(1.2rem, 2vw, 1.5rem); color: #f6cd70; font-weight: 600;"),
                    style="text-align: center; background-color: rgba(255, 255, 255, 0.05); padding: clamp(8px, 2vw, 16px); border-radius: 8px; transition: transform 0.2s; cursor: pointer; &:hover { transform: translateY(-2px); }; min-width: 0;"
                ),
                Div(
                    P("Worst Day", style="margin: 0; color: #999; font-size: clamp(0.8rem, 1.5vw, 0.9rem); font-weight: 500; margin-bottom: 4px;"),
                    P(f"{stats['worst_day']}", style="margin: 0; font-size: clamp(1.2rem, 2vw, 1.5rem); color: #f6cd70; font-weight: 600;"),
                    style="text-align: center; background-color: rgba(255, 255, 255, 0.05); padding: clamp(8px, 2vw, 16px); border-radius: 8px; transition: transform 0.2s; cursor: pointer; &:hover { transform: translateY(-2px); }; min-width: 0;"
                ),
                Div(
                    P("Win Rate", style="margin: 0; color: #999; font-size: clamp(0.8rem, 1.5vw, 0.9rem); font-weight: 500; margin-bottom: 4px;"),
                    P(f"{stats['win_rate']}%", style="margin: 0; font-size: clamp(1.2rem, 2vw, 1.5rem); color: #f6cd70; font-weight: 600;"),
                    style="text-align: center; background-color: rgba(255, 255, 255, 0.05); padding: clamp(8px, 2vw, 16px); border-radius: 8px; transition: transform 0.2s; cursor: pointer; &:hover { transform: translateY(-2px); }; min-width: 0;"
                ),
                Div(
                    P("Avg Position Size", style="margin: 0; color: #999; font-size: clamp(0.8rem, 1.5vw, 0.9rem); font-weight: 500; margin-bottom: 4px;"),
                    P(f"{stats['avg_position_size']:,.0f} Lots", style="margin: 0; font-size: clamp(1.2rem, 2vw, 1.5rem); color: #f6cd70; font-weight: 600;"),
                    style="text-align: center; background-color: rgba(255, 255, 255, 0.05); padding: clamp(8px, 2vw, 16px); border-radius: 8px; transition: transform 0.2s; cursor: pointer; &:hover { transform: translateY(-2px); }; min-width: 0;"
                ),
                Div(
                    P("Risk/Reward", style="margin: 0; color: #999; font-size: clamp(0.8rem, 1.5vw, 0.9rem); font-weight: 500; margin-bottom: 4px;"),
                    P(f"1:{stats['risk_reward']}", style="margin: 0; font-size: clamp(1.2rem, 2vw, 1.5rem); color: #f6cd70; font-weight: 600;"),
                    style="text-align: center; background-color: rgba(255, 255, 255, 0.05); padding: clamp(8px, 2vw, 16px); border-radius: 8px; transition: transform 0.2s; cursor: pointer; &:hover { transform: translateY(-2px); }; min-width: 0;"
                ),
                Div(
                    P("Total Trades", style="margin: 0; color: #999; font-size: clamp(0.8rem, 1.5vw, 0.9rem); font-weight: 500; margin-bottom: 4px;"),
                    P(f"{stats['total_trades']}", style="margin: 0; font-size: clamp(1.2rem, 2vw, 1.5rem); color: #f6cd70; font-weight: 600;"),
                    style="text-align: center; background-color: rgba(255, 255, 255, 0.05); padding: clamp(8px, 2vw, 16px); border-radius: 8px; transition: transform 0.2s; cursor: pointer; &:hover { transform: translateY(-2px); }; min-width: 0;"
                ),
                style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: clamp(12px, 2vw, 20px); margin-top: clamp(16px, 3vw, 24px); width: 100%;"
            ),
            style="background: radial-gradient(circle at top right, rgba(246,205,112, 0.2) 0%, rgba(246,205,112, 0.04) 40%), radial-gradient(circle at bottom left, rgba(246,205,112, 0.2) 0%, rgba(246,205,112, 0.04) 40%); padding: clamp(16px, 3vw, 24px); height: 100%; display: flex; flex-direction: column;"
        ),
        style="background-color: #000; height: 100%; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);"
    )