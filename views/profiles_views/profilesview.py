from fasthtml.common import *
from views.profiles_views.popups.profile_popup import profile_popup

def profiles_view(session):
    return Div(
        Div(
            Div(
                Div(
                    Script(
                        src="/views/profiles_views/gridding/grid_table.js"
                    ),
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
                    class_="loading-spinner",
                    style="""
                        width: 50px; height: 50px; border: 5px solid #f3f3f3;
                        border-top: 5px solid #f6cd70; border-radius: 50%;
                        animation: spin 0.7s linear infinite; 
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        display: none;
                    """,
                    x_show="loading"
                ),
                Div(
                    x_html="createGridTable(profiles)",
                    x_show="!loading",
                    style="width: 100%;"
                ),
                style="width: 100%; overflow: relative;",  # Ensure parent has position: relative

                id="profilesGrid",
                x_data="{ profiles: [], loading: true }",
                x_init="""
                    fetch('/api/profiles/get')
                        .then(res => res.json())
                        .then(data => {
                            profiles = data.profiles;
                            loading = false;
                        })
                """,
            ),
            style="height: 100%; width: 100%; position: relative;"
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