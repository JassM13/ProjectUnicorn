from fasthtml.common import *
def chart_widget():
    return Card(
        Link(rel="stylesheet", href="//cdn.jsdelivr.net/chartist.js/latest/chartist.min.css", type="text/css"),
        Script(src="//cdn.jsdelivr.net/chartist.js/latest/chartist.min.js", _async=False, defer=False),
        Div(
            cls="chart",
            id="chart",
            style="height: 100%; width: 100%; margin: 0; padding: 0;"
        ),
        Script("""
            document.addEventListener('htmx:load', function() {
                var chart = new Chartist.Line('.chart', {
                    labels: [1, 2, 3, 4, 5, 6, 7, 8],
                    series: [
                        [5, 9, 7, 8, 5, 3, 5, 4]
                    ]
                    }, {
                    low: 0,
                    chartPadding: {
                        left: -40,
                        right: 0,
                        top: 20,
                        bottom: -30
                    },
                    lineSmooth: Chartist.Interpolation.simple(),
                    axisX: {
                        showLabel: true,
                        showGrid: true
                    },
                    axisY: {
                        showLabel: true,
                        showGrid: true
                    },
                    showArea: true,
                    fullWidth: true,
                });

                var tooltip = document.createElement('div');
                tooltip.className = 'tooltip';
                document.body.appendChild(tooltip);

                chart.on('draw', function(data) {
                    if(data.type === 'point') {
                        data.element._node.addEventListener('mouseenter', function() {
                            tooltip.style.display = 'block';
                            tooltip.innerText = `Profit: $${data.value.y} \n Date: ${data.axisX.ticks[data.index]}`;
                            var box = data.element._node.getBoundingClientRect();
                            tooltip.style.left = box.left + window.pageXOffset + 'px';
                            tooltip.style.top = box.top + window.pageYOffset - tooltip.offsetHeight + 'px';
                        });

                        data.element._node.addEventListener('mouseleave', function() {
                            tooltip.style.display = 'none';
                        });
                    }
                });
            });
        """),
        Style("""
            .chart .ct-series-a .ct-line {
                stroke: rgba(246, 205, 112, 0.6);
                stroke-width: 3px;
            }
            .chart .ct-series-a .ct-point {
                stroke: rgba(246, 205, 112, 0.6);
                stroke-width: 6px;
            }
            .chart .ct-series-a .ct-area {
                fill: #f6cd70;
            }
            .tooltip {
                position: absolute;
                background-color: #333;
                color: #fff;
                padding: 5px;
                font-size: 12px;
                border-radius: 3px;
                display: none;
                pointer-events: none;
                transform: translate(10%, 0%);
            }
        """),
        style="background-color: #000; height: 100%; overflow: hidden;",
    )