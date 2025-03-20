from fasthtml.common import *

def dashboard_view(session):
    return Div(
        Style("""
            .dashboard-item {
                opacity: 1;  /* Set opacity to 1 directly */
            }
        """),
        Script("""
            document.addEventListener('htmx:load', function() {
                const widgetIds = ['profit', 'progress', 'chart', 'stats'];
                const loadedWidgets = new Set();

                document.addEventListener('htmx:afterSwap', function(evt) {
                    if (evt.detail.target.classList.contains('dashboard-item')) {
                        loadedWidgets.add(evt.detail.target.dataset.swapyItem);
                        if (loadedWidgets.size === widgetIds.length) {
                            document.querySelector('.dashboard-container').classList.add('loaded');
                        }
                    }
                });
            });
        """),
        Div(
            Div(
                Div(
                    Div(cls="dashboard-item", data_swapy_item="profit", hx_get="/api/widgets/profit", hx_trigger="load", hx_swap="innerHTML", style="height: 100%; min-height: min(300px, 30vh);"),
                    cls="dashboard-slot",
                    data_swapy_slot="profit",
                    style="grid-area: profit;"
                ),
                Div(
                    Div(cls="dashboard-item", data_swapy_item="progress", hx_get="/api/widgets/progress", hx_trigger="load", hx_swap="innerHTML", style="height: 100%; min-height: min(300px, 30vh);"),
                    cls="dashboard-slot",
                    data_swapy_slot="progress",
                    style="grid-area: progress;"
                ),
                Div(
                    Div(cls="dashboard-item", data_swapy_item="chart", hx_get="/api/widgets/chart", hx_trigger="load", hx_swap="innerHTML", style="height: 100%; min-height: min(300px, 30vh);"),
                    cls="dashboard-slot",
                    data_swapy_slot="chart",
                    style="grid-area: chart;"
                ),
                Div(
                    Div(cls="dashboard-item", data_swapy_item="stats", hx_get="/api/widgets/stats", hx_trigger="load", hx_swap="innerHTML", style="height: 100%; min-height: min(300px, 30vh);"),
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
                        overflow: hidden;
                    """
            ),
            style="""
                    color: white; height: 95vh; max-height: 95vh; border-radius: 16px;
                    position: absolute; right: 24px; top: 20px;
                    left: 100px; bottom: 24px; overflow: hidden;
                """
        )
    )
