from fasthtml.common import *
from fh_plotly import plotly2fasthtml
import plotly.express as px
import pandas as pd
import numpy as np

def generate_line_chart():
    data = [3590, 5239, 3239, 2390, 1239, 5239]
    df = pd.DataFrame({
        'Day': [f'Day {i+1}' for i in range(len(data))],
        'Profit': data
    })
    fig = px.line(df, x='Day', y='Profit',
                  line_shape='spline',
                  template='plotly_dark')
    fig.update_traces(
        line_color='rgba(246, 205, 112, 0.6)',
        fill='tozeroy',
        fillcolor='rgba(246, 205, 112, 0.2)',
        mode='lines+markers',
        hovertemplate='Profit: $%{y}<br>Day: %{x}'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=40),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        #autosize=True
    )
    fig.update_layout(
        modebar_remove=[
            'toImage', 'zoom', 'pan', 
            'select', 'lasso2d', 'zoomIn2d', 
            'zoomOut2d', 'autoScale2d', 'resetScale2d'
        ],
        dragmode='pan',
        yaxis_title_text=None,
        xaxis_title_text=None,
        yaxis_fixedrange=True,
        xaxis_fixedrange=True,
        )
    return fig

def chart_widget():
    return Card(
        Div(
            plotly2fasthtml(generate_line_chart()),
            id="chart-container",
            hx_get="/api/chart/refresh",
            hx_trigger="resize from:window",
            hx_swap="innerHTML"
        ),
        style="background-color: #000; height: 100%; overflow: hidden;"
    )