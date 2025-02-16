from fasthtml.common import *

def progress_widget():
    return Card(
        Div(
            H2("Trading Goals", style="margin: 0; color: #f6cd70;"),
            Div(
                Div(
                    P("Monthly Target", style="margin: 0;"),
                    Div(
                        Div(
                            style="width: 75%; height: 100%; background-color: #4CAF50; border-radius: 8px;"
                        ),
                        style="width: 100%; height: 10px; background-color: #333; border-radius: 8px; margin: 5px 0;"
                    ),
                    P("$30,000 / $40,000", style="margin: 0; color: #f6cd70;"),
                    style="margin-bottom: 15px;"
                ),
                Div(
                    P("Win Streak Goal", style="margin: 0;"),
                    Div(
                        Div(
                            style="width: 60%; height: 100%; background-color: #f6cd70; border-radius: 8px;"
                        ),
                        style="width: 100%; height: 10px; background-color: #333; border-radius: 8px; margin: 5px 0;"
                    ),
                    P("6 / 10 Trades", style="margin: 0; color: #f6cd70;"),
                    style="margin-bottom: 15px;"
                ),
                Div(
                    P("Risk Management", style="margin: 0;"),
                    Div(
                        Div(
                            style="width: 90%; height: 100%; background-color: #4CAF50; border-radius: 8px;"
                        ),
                        style="width: 100%; height: 10px; background-color: #333; border-radius: 8px; margin: 5px 0;"
                    ),
                    P("90% Compliance", style="margin: 0; color: #f6cd70;"),
                ),
                style="padding: 10px 0;"
            ),
            style="padding: 20px;"
        ),
        style="background-color: #222; border-radius: 12px; height: 100%;"
    )