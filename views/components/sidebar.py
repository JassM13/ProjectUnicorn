from fasthtml.common import *
from fa6_icons import svgs

def sidebar(active):
    return Div(
        # Rounded Sidebar container
        Div(
            # Logo at the top
            Div(
                A(Img(src="/assets/logo.png", alt="ProjectUnicorn Logo", width="50px", height="50px", style="border-radius: 10%;"), href="/"),
                style="text-align:center; margin: 20px 0;"
            ),

            # Navigation icons in the middle (highlight based on active view)
            Div(
                A(I(svgs.house.solid), href="/dashboard", style=f"display:block; margin: 20px 0; text-align:center; font-size:16px; color:white; opacity: {'1' if active == 'dashboard' else '0.5'};"),
                A(I(svgs.message.solid), href="/chat", style=f"display:block; margin: 20px 0; text-align:center; font-size:16px; color:white; opacity: {'1' if active == 'chat' else '0.5'};"),
                A(I(svgs.gear.solid), href="/settings", style=f"display:block; margin: 20px 0; text-align:center; font-size:16px; color:white; opacity: {'1' if active == 'settings' else '0.5'};"),
            ),

            # User Profile at the bottom
            Div(
                A(I(svgs.user.solid), href="/profile", style="display:block; margin-top: 20px 0; text-align:center; font-size:16px; color:white; opacity: 1;"),
            ),

            # Sidebar Styling
            style="""
                width: 70px; height: 95vh; background-color: #000; position: fixed; 
                top: 20px; left: 20px; display: flex; flex-direction: column; 
                justify-content: space-between; color:white; 
                padding: 20px; border-radius: 16px; 
            """
        )
    )
