from fasthtml.common import *

def stats_widget():
    return Card(
        Div(
            H2("Trading Stats", style="margin: 0; color: #f6cd70;"),
            Div(
                Div(
                    P("Win Rate", style="margin: 0;"),
                    P("68.5%", style="margin: 0; font-size: 1.25rem; color: #4CAF50;"),
                    style="text-align: center;"
                ),
                Div(
                    P("Avg Position Size", style="margin: 0;"),
                    P("$5,420", style="margin: 0; font-size: 1.25rem; color: #f6cd70;"),
                    style="text-align: center;"
                ),
                Div(
                    P("Risk/Reward", style="margin: 0;"),
                    P("1:2.5", style="margin: 0; font-size: 1.25rem; color: #f6cd70;"),
                    style="text-align: center;"
                ),
                Div(
                    P("Total Trades", style="margin: 0;"),
                    P("142", style="margin: 0; font-size: 1.25rem; color: #f6cd70;"),
                    style="text-align: center;"
                ),
                style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px;"
            ),
            style="padding: 16px;"
        ),
        style="background-color: #222; border-radius: 12px; height: 100%;"
    )