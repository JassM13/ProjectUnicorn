from fasthtml.common import *
from views.profiles_views.popups.profile_popup import profile_popup
from views.profiles_views.gridding.grid_table import create_grid_table

def profiles_view(session):
    return Div(
        Div(
            Div(
                H2("Profiles", style="margin: 0 0 4px 0;"),
                Button(
                    Img(src='assets/svgs/User/User_Add.svg', style="margin-right: 8px;"),
                    "New Profile",
                    id="add_profile_button",
                    style="""background-color: #f6cd70; color: black; border: none; 
                           border-radius: 16px; padding: 8px 16px; font-size: 14px; 
                           font-weight: 600; cursor: pointer; margin-bottom: 8px;
                           display: flex; align-items: center; justify-content: center;""",
                    hx_on_click="document.getElementById('profile_modal_overlay').classList.add('show'); document.getElementById('profile_form_container').classList.add('show');"
                ),
                style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px;"
            ),
            Div(
                Div(
                    Div(style="width: 50px; height: 50px; border: 5px solid rgba(255, 255, 255, 0.1); border-top: 5px solid #ffffff; border-radius: 50%; animation: spin 1s linear infinite; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);"),
                    style="display: flex; position: absolute; top: 0; left: 0; width: 100%; height: 100%; justify-content: center; align-items: center; z-index: 1; border-radius: 8px; pointer-events: none;",
                    id="loading-overlay"
                ),
                id="profilesGrid",
                hx_get="/api/profiles/get",
                hx_trigger="load, profileCreated from:body, profileDeleted from:body",
                hx_swap="innerHTML"
            ),
            style="width: 100%;"
        ),
        profile_popup(),
        Style("""
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        """),
        style="""
            display: flex; flex-direction: column; padding: 30px; color: white;
            height: 95vh; border-radius: 16px; position: absolute; right: 20px;
            top: 20px; left: 100px; bottom: 20px; overflow: auto;
            border: 1px solid rgba(255, 255, 255, 0.1);
        """
    )