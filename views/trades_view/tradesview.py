from fasthtml.common import *
from views.profiles_views.popups.profile_popup import profile_popup
from views.trades_view.popups.trade_popup import trade_popup

def trades_view(session=None):
    return Div(
        Div(
            Script(src="/views/trades_view/js/dropdown.js", defer=True),
            Script(src="static/js/alpine.min.js", defer=True),
            Div(
                H2("Trades", style="margin: 0 0 8px 0;"),
                Div(
                    Div(
                        Div(
                            x_html="await createDropdown(profiles)",
                            style="display: flex; align-items: center;"
                        ),
                        style="display: flex; align-items: center;",
                        x_data="{ profiles: [], selectedProfile: '', loading: true }",
                        x_init="""
                            fetch('/api/profiles/get')
                                .then(res => res.json())
                                .then(data => {
                                    profiles = data.profiles;
                                    loading = false;
                            });
                        """,
                    ),
                    Button(
                        Img(src='assets/svgs/User/User_Add.svg', style="margin-right: 8px;"),
                        "New Trade",
                        id="add_trade_button",
                        style="""background-color: #f6cd70; color: black; border: none; 
                               border-radius: 16px; padding: 8px 16px; font-size: 14px; 
                               font-weight: 600; cursor: pointer; margin-left: auto;
                               display: flex; align-items: center; justify-content: center;""",
                        hx_on_click="document.getElementById('trade_modal_overlay').classList.add('show'); document.getElementById('trade_form_container').classList.add('show');"
                    ),
                    style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px;"
                ),
            Div(id="tradesGrid", cls="custom-grid"),
            style="width: 100%;"
        ),
        trade_popup(),
        # No script tag needed - using inline hx_on_click instead
        style="""
            display: flex; flex-direction: column; padding: 30px;
            color: white; height: 95vh; border-radius: 16px;
            position: absolute; right: 20px; top: 20px; left: 100px;
            bottom: 20px; overflow: auto; border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        """
    )
    )