from fasthtml.common import *

def stats_widget():
    return Card(
        Div(
            H2("Trading Stats", style="margin: 0; color: #f6cd70; font-size: clamp(1.2rem, 2vw, 1.75rem); font-weight: 600; letter-spacing: 0.5px;"),
            Div(
                Div(
                    P("Win Rate", style="margin: 0; color: #999; font-size: clamp(0.8rem, 1.5vw, 0.9rem); font-weight: 500; margin-bottom: 4px;"),
                    P("68.5%", style="margin: 0; font-size: clamp(1.2rem, 2vw, 1.5rem); color: #4CAF50; font-weight: 600;"),
                    style="text-align: center; background-color: #2a2a2a; padding: clamp(8px, 2vw, 16px); border-radius: 8px; transition: transform 0.2s; cursor: pointer; &:hover { transform: translateY(-2px); }; min-width: 0;"
                ),
                Div(
                    P("Avg Position Size", style="margin: 0; color: #999; font-size: clamp(0.8rem, 1.5vw, 0.9rem); font-weight: 500; margin-bottom: 4px;"),
                    P("$5,420", style="margin: 0; font-size: clamp(1.2rem, 2vw, 1.5rem); color: #f6cd70; font-weight: 600;"),
                    style="text-align: center; background-color: #2a2a2a; padding: clamp(8px, 2vw, 16px); border-radius: 8px; transition: transform 0.2s; cursor: pointer; &:hover { transform: translateY(-2px); }; min-width: 0;"
                ),
                Div(
                    P("Risk/Reward", style="margin: 0; color: #999; font-size: clamp(0.8rem, 1.5vw, 0.9rem); font-weight: 500; margin-bottom: 4px;"),
                    P("1:2.5", style="margin: 0; font-size: clamp(1.2rem, 2vw, 1.5rem); color: #f6cd70; font-weight: 600;"),
                    style="text-align: center; background-color: #2a2a2a; padding: clamp(8px, 2vw, 16px); border-radius: 8px; transition: transform 0.2s; cursor: pointer; &:hover { transform: translateY(-2px); }; min-width: 0;"
                ),
                Div(
                    P("Total Trades", style="margin: 0; color: #999; font-size: clamp(0.8rem, 1.5vw, 0.9rem); font-weight: 500; margin-bottom: 4px;"),
                    P("142", style="margin: 0; font-size: clamp(1.2rem, 2vw, 1.5rem); color: #f6cd70; font-weight: 600;"),
                    style="text-align: center; background-color: #2a2a2a; padding: clamp(8px, 2vw, 16px); border-radius: 8px; transition: transform 0.2s; cursor: pointer; &:hover { transform: translateY(-2px); }; min-width: 0;"
                ),
                style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: clamp(12px, 2vw, 20px); margin-top: clamp(16px, 3vw, 24px); width: 100%;"
            ),
            style="padding: clamp(16px, 3vw, 24px); height: 100%; display: flex; flex-direction: column;"
        ),
        style="background-color: #222; border-radius: 16px; height: 100%; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);"
    )