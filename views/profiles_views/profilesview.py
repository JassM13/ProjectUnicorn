from fasthtml.common import *
from views.profiles_views.popups.profile_popup import profile_popup
from views.profiles_views.gridding.grid_table import create_grid_table

def profiles_view(session=None):
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
                           display: flex; align-items: center; justify-content: center;"""
                ),
                style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px;"
            ),
            # Instead of embedding the create_grid_table directly, use a Div with hx_get
            # This will automatically load the grid with the data
            Div(
                # Display a loading message until HTMX loads the data
                Div("Loading profiles...", style="text-align: center; padding: 20px;"),
                # Add a loading indicator
                Div(
                    Div(style="width: 50px; height: 50px; border: 5px solid #f3f3f3; border-top: 5px solid #f6cd70; border-radius: 50%; animation: spin 1s linear infinite;"),
                    style="display: none; position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.5); justify-content: center; align-items: center; z-index: 1000;",
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
        # Toggle popup script
        Script("""
            function closeProfileModal() {
                document.getElementById('profile_modal_overlay').style.opacity = '0';
                document.getElementById('profile_modal_overlay').style.visibility = 'hidden';
                document.querySelector('#modal_content').style.opacity = '0';
                document.querySelector('#modal_content').style.visibility = 'hidden';
                document.querySelector('#modal_content').style.transform = 'translate(-50%, -50%) scale(0.8)';
                document.getElementById('profile_form').reset();
            }

            document.addEventListener('htmx:load', function() {
                const addButton = document.getElementById('add_profile_button');
                if (addButton) {
                    addButton.addEventListener('click', function() {
                        document.getElementById('profile_modal_overlay').style.opacity = '1';
                        document.getElementById('profile_modal_overlay').style.visibility = 'visible';
                        document.querySelector('#modal_content').style.opacity = '1';
                        document.querySelector('#modal_content').style.visibility = 'visible';
                        document.querySelector('#modal_content').style.transform = 'translate(-50%, -50%) scale(1)';
                    });
                }
            });
        """),
        # Add CSS for spinner animation
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