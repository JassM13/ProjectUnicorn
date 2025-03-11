from fasthtml.common import *
import json
from views.profiles_views.popups.profile_popup import profile_popup
from views.profiles_views.gridding.grid_table import create_grid_table

def profiles_view(session=None):
    # Initialize with empty profiles array
    # Profiles will be loaded via HTMX from the API endpoint
    
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
                id="profilesGrid",
                hx_get="/api/profiles/get",
                hx_trigger="load",
                hx_swap="innerHTML"
            ),
            style="width: 100%;"
        ),
        profile_popup(),
        # Modal event handling script
        Script("""
            function showProfileModal() {
                const modalOverlay = document.getElementById('profile_modal_overlay');
                const modal = document.querySelector('#profile_popup_container > div:first-child');
                
                if (!modalOverlay || !modal) return;
                
                modalOverlay.style.opacity = '1';
                modalOverlay.style.visibility = 'visible';
                modal.style.opacity = '1';
                modal.style.visibility = 'visible';
                modal.style.transform = 'translate(-50%, -50%) scale(1)';
            }
            
            function hideProfileModal() {
                const modalOverlay = document.getElementById('profile_modal_overlay');
                const modal = document.querySelector('#profile_popup_container > div:first-child');
                const profileForm = document.getElementById('profile_form');
                
                if (!modalOverlay || !modal) return;
                
                modalOverlay.style.opacity = '0';
                modalOverlay.style.visibility = 'hidden';
                modal.style.opacity = '0';
                modal.style.visibility = 'hidden';
                modal.style.transform = 'translate(-50%, -50%) scale(0.8)';
                if (profileForm) profileForm.reset();
            }
            
            // Set up event listeners when the document is loaded
            document.addEventListener('htmx:load', function() {
                // Setup modal event listeners
                const addButton = document.getElementById('add_profile_button');
                const cancelButton = document.getElementById('cancel_profile_button');
                const modalOverlay = document.getElementById('profile_modal_overlay');
                
                if (addButton) addButton.addEventListener('click', showProfileModal);
                if (cancelButton) cancelButton.addEventListener('click', hideProfileModal);
                if (modalOverlay) modalOverlay.addEventListener('click', function(e) {
                    if (e.target === modalOverlay) hideProfileModal();
                });
            });
            
            // Refresh profiles after successful profile creation
            document.addEventListener('htmx:afterRequest', function(event) {
                if (event.detail.target && event.detail.target.id === 'profile_form_alert') {
                    // Check if the response indicates success
                    if (event.detail.xhr.responseText.includes('successfully')) {
                        // Hide modal after successful profile creation
                        setTimeout(function() {
                            hideProfileModal();
                            // Refresh profiles by triggering a GET request on the profilesGrid
                            htmx.trigger('#profilesGrid', 'htmx:refresh');
                        }, 1500); // Short delay to allow user to see success message
                    }
                }
            });
        """),
        style="""
            display: flex;
            flex-direction: column;
            padding: 30px;
            background-color: #090909;
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