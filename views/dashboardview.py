from fasthtml.common import *
from views.components.widgets.total_profit import total_profit_widget
from views.components.widgets.chart import chart_widget
from views.components.widgets.stats import stats_widget
from views.components.widgets.progress import progress_widget

def dashboard_view():
    return Div(
        Script(src="https://unpkg.com/swapy/dist/swapy.min.js"),
        Script("""
            document.addEventListener('DOMContentLoaded', function() {
                const container = document.querySelector('.dashboard-container');
                window.swapy = Swapy.createSwapy(container, {
                    draggable: true,
                    animation: 'dynamic'
                });

                // Add keyboard shortcut to toggle Swapy (Ctrl/Cmd + Shift + S)
                document.addEventListener('keydown', function(e) {
                    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'S') {
                        e.preventDefault();
                        window.swapy.enable(!window.swapy.isEnabled());
                    }
                });
            });
        """),
        Div(
            Div(
                Div(
                    Div(total_profit_widget(), cls="dashboard-item", data_swapy_item="profit", style="height: 100%; min-height: min(300px, 30vh);"),
                    cls="dashboard-slot",
                    data_swapy_slot="profit",
                    style="grid-area: profit;"
                ),
                Div(
                    Div(progress_widget(), cls="dashboard-item", data_swapy_item="progress", style="height: 100%; min-height: min(300px, 30vh);"),
                    cls="dashboard-slot",
                    data_swapy_slot="progress",
                    style="grid-area: progress;"
                ),
                Div(
                    Div(chart_widget(), cls="dashboard-item", data_swapy_item="chart", style="height: 100%; min-height: min(300px, 30vh);"),
                    cls="dashboard-slot",
                    data_swapy_slot="chart",
                    style="grid-area: chart;"
                ),
                Div(
                    Div(stats_widget(), cls="dashboard-item", data_swapy_item="stats", style="height: 100%; min-height: min(300px, 30vh);"),
                    cls="dashboard-slot",
                    data_swapy_slot="stats",
                    style="grid-area: stats;"
                ),
                cls="dashboard-container",
                style="""display: grid; gap: 12px;
                        grid-template-areas:
                            'profit progress'
                            'chart stats';
                        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
                        grid-template-rows: minmax(min(300px, 30vh), 1fr) minmax(min(300px, 30vh), 1fr);
                        height: 100%;
                        width: 100%;
                        border-radius: 16px;
                        overflow: hidden;
                    """
            ),
            style="""padding: 24px; background-color: #000;
                    color: white; height: 95vh; max-height: 95vh; border-radius: 16px;
                    position: absolute; right: 24px; top: 20px;
                    left: 100px; bottom: 24px; overflow: hidden;
                """
        )
    )
