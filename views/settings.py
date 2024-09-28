from fasthtml.common import *

def settings_view():
    return Div(
        H1("Settings", style="text-align:center;"),
        P("Adjust your preferences here.", style="text-align:center;"),
        style=""" padding: 20px; background-color: #000; 
                    color: white; height: 95vh; border-radius: 16px; 
                    display: flex; flex-direction: column; justify-content: center; 
                    align-items: center; position: absolute; right: 20px; top: 20px; 
                    left: 100px; bottom: 20px;
                """
    )
