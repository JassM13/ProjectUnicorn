from fasthtml.common import *
import json
from views.profiles_views.popups.profile_popup import profile_popup

def trades_view(session=None):
    # Initialize with empty profiles array
    # Profiles will be loaded via HTMX from the API endpoint
    profiles = []
    
    return Div(
        Link(rel="stylesheet", href="/views/profiles_views/gridding/styles.css"),
        Script(src="/views/profiles_views/gridding/grid.js"),
        Div(
            Div(
                H2("Trades", style="margin: 0 0 8px 0;"),
                Button(
                    Img(src='assets/svgs/User/User_Add.svg', style="margin-right: 8px;"),
                    "New Trade",
                    id="add_trade_button",
                    style="""background-color: #f6cd70; color: black; border: none; 
                           border-radius: 16px; padding: 8px 16px; font-size: 14px; 
                           font-weight: 600; cursor: pointer; margin-bottom: 8px;
                           display: flex; align-items: center; justify-content: center;"""
                ),
                style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px;"
            ),
            Div(id="tradesGrid", cls="custom-grid"),
            style="width: 100%;"
        ),
        profile_popup(),
        style="""
            display: flex;
            flex-direction: column;
            padding: 30px;
            background-color: #000;
            color: white;
            height: 95vh;
            border-radius: 16px;
            position: absolute;
            right: 20px;
            top: 20px;
            left: 100px;
            bottom: 20px;
            overflow: auto;
        """
    )