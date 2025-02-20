from fasthtml.common import *

def progress_widget():
    return Card(
        Div(
            H2("Trading Goals", style="margin: 0; color: rgba(246,205,112, 0.6); font-size: clamp(1.2rem, 2vw, 1.5rem); flex: 0 0 auto;"),
            Div(
                Div(
                    P("Monthly Target", style="margin: 0; font-size: clamp(0.8rem, 1.5vw, 0.9rem); flex: 0 0 auto;"),
                    Div(
                        Div(
                            style="width: 75%; height: 100%; background-color: rgba(246,205,112, 0.6); border-radius: 8px;"
                        ),
                        style="width: 100%; height: min(8px, 1.5vh); background-color: #333; border-radius: 8px; margin: 0.5vh 0;"
                    ),
                    P("$30,000 / $40,000", style="margin: 0; color: rgba(246,205,112, 0.6); font-size: clamp(0.9rem, 1.8vw, 1.1rem); flex: 0 0 auto;"),
                    style="margin-bottom: min(12px, 2vh); flex: 1; display: flex; flex-direction: column; justify-content: space-between;"
                ),
                Div(
                    P("Win Streak Goal", style="margin: 0; font-size: clamp(0.8rem, 1.5vw, 0.9rem); flex: 0 0 auto;"),
                    Div(
                        Div(
                            style="width: 60%; height: 100%; background-color: rgba(246,205,112, 0.6); border-radius: 8px;"
                        ),
                        style="width: 100%; height: min(8px, 1.5vh); background-color: #333; border-radius: 8px; margin: 0.5vh 0;"
                    ),
                    P("6 / 10 Trades", style="margin: 0; color: rgba(246,205,112, 0.6); font-size: clamp(0.9rem, 1.8vw, 1.1rem); flex: 0 0 auto;"),
                    style="margin-bottom: min(12px, 2vh); flex: 1; display: flex; flex-direction: column; justify-content: space-between;"
                ),
                Div(
                    P("Risk Management", style="margin: 0; font-size: clamp(0.8rem, 1.5vw, 0.9rem); flex: 0 0 auto;"),
                    Div(
                        Div(
                            style="width: 90%; height: 100%; background-color: rgba(246,205,112, 0.6); border-radius: 8px;"
                        ),
                        style="width: 100%; height: min(8px, 1.5vh); background-color: #333; border-radius: 8px; margin: 0.5vh 0;"
                    ),
                    P("90% Compliance", style="margin: 0; color: rgba(246,205,112, 0.6); font-size: clamp(0.9rem, 1.8vw, 1.1rem); flex: 0 0 auto;"),
                    style="flex: 1; display: flex; flex-direction: column; justify-content: space-between;"
                ),
                style="padding: min(8px, 1.5vh) 0; flex: 1; display: flex; flex-direction: column; justify-content: space-between;"
            ),
            style="padding: min(16px, 3vh) min(24px, 4vw); height: 100%; display: flex; flex-direction: column;"
        ),
        style="background-color: #000; height: 100%; overflow: hidden;"
    )