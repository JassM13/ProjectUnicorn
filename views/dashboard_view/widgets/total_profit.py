from fasthtml.common import *

def total_profit_widget():
    return Card(
        Div(
            Div(
                P("$24,680.50", style="font-size: clamp(1.6rem, 3vw, 2rem); margin: 12px 0; color: #00E676; font-weight: 700; text-shadow: 0 2px 4px rgba(0, 230, 118, 0.2);"),
                P("+12.5% this month", style="color: #00E676; margin: 0; font-size: clamp(0.8rem, 1.5vw, 0.9rem); font-weight: 500; letter-spacing: 0.3px;"),
                style="text-align: center; min-height: 0; background: linear-gradient(180deg, rgba(0, 230, 118, 0.1) 0%, rgba(34, 34, 34, 0) 100%); padding: 16px; border-radius: 8px;"
            ),
            Div(
                Div(
                    P("Daily Profit", style="margin: 0 0 8px 0; font-size: clamp(0.8rem, 1.5vw, 0.9rem); color: #999; font-weight: 500;"),
                    P("$420.30", style="margin: 0; color: #00E676; font-size: clamp(1rem, 2vw, 1.2rem); font-weight: 600;"),
                    style="text-align: center; flex: 1; min-width: 0; padding: 16px; background: rgba(255, 255, 255, 0.05); border-radius: 8px; transition: transform 0.2s ease-in-out; cursor: pointer; &:hover { transform: translateY(-2px); }"
                ),
                Div(
                    P("Weekly Profit", style="margin: 0 0 8px 0; font-size: clamp(0.8rem, 1.5vw, 0.9rem); color: #999; font-weight: 500;"),
                    P("$2,850.40", style="margin: 0; color: #00E676; font-size: clamp(1rem, 2vw, 1.2rem); font-weight: 600;"),
                    style="text-align: center; flex: 1; min-width: 0; padding: 16px; background: rgba(255, 255, 255, 0.05); border-radius: 8px; transition: transform 0.2s ease-in-out; cursor: pointer; &:hover { transform: translateY(-2px); }"
                ),
                style="padding: 20px; display: flex; justify-content: space-around; margin-top: 16px; gap: 16px; flex-wrap: wrap;"
            ),
            style="height: 100%; display: flex; flex-direction: column; justify-content: space-between; background: linear-gradient(135deg, rgba(255, 0, 0, 0.01) 0%, rgba(255, 0, 0, 0.1) 100%);"
        ),
        style="background-color: #000; height: 100%; overflow: hidden; box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2); transition: transform 0.3s ease-in-out; &:hover { transform: translateY(-4px); }"
    )