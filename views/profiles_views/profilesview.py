from fasthtml.common import *

def profiles_view():
    return Div(
        Div(
            Div(
                H2("Profiles", style="margin: 0;"),
            ),
            style="display: flex; height: 100%; position: relative;"
        ),
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
            overflow: hidden;
        """
    )