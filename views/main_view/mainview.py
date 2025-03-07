from fasthtml.common import *
from views.dashboard_view.dashboardview import dashboard_view
from views.pages.tradesview import trades_view
from views.pages.calendarview import calendar_view
from views.profiles_views.profilesview import profiles_view

def sidebar(active):
    return Div(
        # Rounded Sidebar container
        Div(
            # Logo at the top
            Div(
                A(Img(src="/assets/favicon.svg", alt="ProjectUnicorn Logo", width="50px", height="50px", style="border-radius: 10%;"), href="/"),
                style="text-align:center; margin: 20px 0;"
            ),

            Div(
                A(Img(src="/assets/svgs/Navigation/House_01.svg", alt="Dashboard"), 
                  hx_get="/dashboard", hx_target="#main-content", hx_push_url="true",
                  style=f"display:block; margin: 20px 0; text-align:center; font-size:16px; color:white; opacity: {'1' if active == 'dashboard' else '0.5'};"),
                A(Img(src="/assets/svgs/File/Notebook.svg", alt="Chat"), 
                  hx_get="/trades", hx_target="#main-content", hx_push_url="true",
                  style=f"display:block; margin: 20px 0; text-align:center; font-size:16px; color:white; opacity: {'1' if active == 'trades' else '0.5'};"),
                A(Img(src="/assets/svgs/Calendar/Calendar_Days.svg", alt="Calendar"), 
                  hx_get="/calendar", hx_target="#main-content", hx_push_url="true",
                  style=f"display:block; margin: 20px 0; text-align:center; font-size:16px; color:white; opacity: {'1' if active == 'calendar' else '0.5'};"),
                A(Img(src="/assets/svgs/User/Users_Group.svg", alt="Profiles"), 
                  hx_get="/profiles", hx_target="#main-content", hx_push_url="true",
                  style=f"display:block; margin: 20px 0; text-align:center; font-size:16px; color:white; opacity: {'1' if active == 'profiles' else '0.5'};"),
            ),

            # User Profile at the bottom
            Div(
                A(Img(src="/assets/svgs/User/User_Circle.svg", alt="Account"), href="/account", style="display:block; margin-top: 20px 0; text-align:center; font-size:16px; color:white; opacity: 1;"),
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

def mainview(active="dashboard"):
    view_map = {
        "dashboard": dashboard_view,
        "trades": trades_view,
        "calendar": calendar_view,
        "profiles": profiles_view
    }
    
    # Get the appropriate view function or default to dashboard
    view_func = view_map.get(active, dashboard_view)
    
    return (Title("Hello"), Div(
            sidebar(active=active),
            Div(
                Div(
                    Div(
                        style="""background-color: #000; height: 95vh; max-height: 95vh; border-radius: 16px;
                            position: absolute; right: 24px; top: 20px;
                            left: 100px; bottom: 24px; overflow: hidden;
                        """
                    ),
                    view_func(),
                    id="main-content"
                ),
                style="margin-left: 100px;"
            ),
            style="display:flex;"
        )
    )