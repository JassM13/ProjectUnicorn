from fasthtml.common import *

def total_profit_widget():
    return Card(
        Div(
            H2("Total Profit", style="margin: 0; color: #f6cd70; font-size: 1.5rem;"),
            Div(
                P("$24,680.50", style="font-size: 1.75rem; margin: 8px 0; color: #4CAF50;"),
                P("+12.5% this month", style="color: #4CAF50; margin: 0; font-size: 0.9rem;"),
                style="text-align: center;"
            ),
            Div(
                Div(
                    P("Daily Profit", style="margin: 0; font-size: 0.9rem;"),
                    P("$420.30", style="margin: 0; color: #4CAF50;"),
                    style="text-align: center;"
                ),
                Div(
                    P("Weekly Profit", style="margin: 0; font-size: 0.9rem;"),
                    P("$2,850.40", style="margin: 0; color: #4CAF50;"),
                    style="text-align: center;"
                ),
                style="display: flex; justify-content: space-around; margin-top: 12px;"
            ),
            style="padding: 16px;"
        ),
        style="background-color: #222; border-radius: 12px; height: 100%;"
    )