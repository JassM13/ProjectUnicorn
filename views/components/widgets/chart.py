from fasthtml.common import *

data = [3590, 5239, 3239, 2390, 1239, 5239]

def chart_widget():
    return Card(
        Div(
            Script(src="https://cdn.jsdelivr.net/npm/chart.js"),
            Div(
                Canvas(id="profitChart"),
                style="height: 105%; margin: -10px;"
            ),
            Script(f"""
                document.addEventListener('DOMContentLoaded', function() {{
                    const ctx = document.getElementById('profitChart').getContext('2d');
                    
                    const data = {{
                        labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6'],
                        datasets: [{{
                            label: 'Daily Profit',
                            data: {data},
                            fill: true,
                            backgroundColor: 'rgba(246, 205, 112, 0.2)',
                            borderColor: 'rgba(246,205,112, 0.6)',
                            tension: 0.4
                        }}]
                    }};

                    const config = {{
                        type: 'line',
                        data: data,
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            animation: false,
                            plugins: {{
                                legend: {{
                                    display: false
                                }}
                            }},
                            scales: {{
                                y: {{
                                    display: false,
                                    beginAtZero: true,
                                    grid: {{
                                        display: false,
                                        drawBorder: false
                                    }}
                                }},
                                x: {{
                                    display: false,
                                    grid: {{
                                        display: false,
                                        drawBorder: false
                                    }}
                                }}
                            }},
                            layout: {{
                                padding: 0
                            }}
                        }}
                    }};

                    new Chart(ctx, config);
                }});
            """),
            style="height: 100%; width: 100%;"
        ),
        style="background-color: #000; height: 100%; overflow: hidden;"
    )