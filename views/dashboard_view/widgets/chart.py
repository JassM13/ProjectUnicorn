from fasthtml.common import *

data = [3590, 5239, 3239, 2390, 1239, 5239]

def chart_widget():
    return Card(
        Div(
            Script(src="/views/dashboard_view/widgets/js/chart.umd.min.js"),
            Div(
                Canvas(id="profitChart"),
                style="height: 105%; margin: -10px;"
            ),
            Script(f"""
                // Use window-scoped variable to prevent redeclaration errors
                if (typeof window.profitChartInstance === 'undefined') {{
                    window.profitChartInstance = null;
                }}
                
                // Also use window-scoped observer to prevent redeclaration errors
                if (typeof window.profitChartObserver === 'undefined') {{
                    window.profitChartObserver = null;
                }}
                
                function initializeChart() {{
                    const canvas = document.getElementById('profitChart');
                    if (!canvas) return;
                    
                    const ctx = canvas.getContext('2d');
                    if (window.profitChartInstance) {{
                        window.profitChartInstance.destroy();
                    }}
                    
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

                    window.profitChartInstance = new Chart(ctx, config);
                }}

                // Initialize chart when HTMX loads content
                document.addEventListener('htmx:load', function() {{
                    // Add a small delay to ensure canvas is available
                    setTimeout(initializeChart, 100);
                }});

                // Initialize chart when element becomes visible
                if (window.profitChartObserver) {{
                    window.profitChartObserver.disconnect();
                }}
                
                window.profitChartObserver = new MutationObserver((mutations) => {{
                    mutations.forEach((mutation) => {{
                        if (mutation.type === 'childList' && document.getElementById('profitChart')) {{
                            initializeChart();
                        }}
                    }});
                }});

                // Start observing the document for DOM changes
                window.profitChartObserver.observe(document.documentElement, {{
                    childList: true,
                    subtree: true
                }});
            """),
            style="height: 100%; width: 100%;"
        ),
        style="background-color: #000; height: 100%; overflow: hidden;"
    )