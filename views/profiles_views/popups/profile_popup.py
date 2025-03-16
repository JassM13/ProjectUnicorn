from fasthtml.common import *

def profile_popup():
    return Div(
        # Modal overlay with HTMX animation classes
        Div(
            id="profile_modal_overlay",
            cls="profile-modal-overlay",
            hx_on_click="this.classList.remove('show'); document.getElementById('profile_form_container').classList.remove('show');"
        ),
        
        # Profile Form Container
        Div(
            Div(
                H2("Create New Profile", style="margin: 0 0 20px 0; color: #f6cd70;"),
                Form(
                    Input(type="text", 
                         name="profile_name", 
                         placeholder="Profile Name", 
                         required=True,
                         cls="profile-input"
                    ),
                    
                    Div(
                        Label(
                            Input(
                                type="checkbox",
                                name="broker_account",
                                disabled=True,
                                style="margin-right: 8px; vertical-align: middle;"
                            ),
                            Span(
                                "Connect to Broker Account ",
                                Span(
                                    "(unavailable)",
                                    style="color: #ff6b6b; font-size: 12px; opacity: 0.8;"
                                ),
                                style="vertical-align: middle;"
                            ),
                            style="display: inline-flex; align-items: center; color: white; user-select: none;"
                        ),
                        style="margin: 16px 0px 16px 0px; padding: 4px 0;"
                    ),
        
                    # Response message area
                    Div(
                        id="profile_form_alert",
                        cls="profile-response"
                    ),
                    
                    # Action Buttons
                    Div(
                        Button("Cancel",
                            type="button",
                            cls="profile-btn cancel-btn",
                            hx_on_click="document.getElementById('profile_modal_overlay').classList.remove('show'); document.getElementById('profile_form_container').classList.remove('show'); this.form.reset();"
                        ),
                        Button("Create Profile",
                            type="submit",
                            cls="profile-btn submit-btn"
                        ),
                        style="display: flex; justify-content: flex-end; gap: 12px;"
                    ),
                    
                    id="profile_form",
                    style="width: 100%;",
                    hx_post="/api/profiles",
                    hx_target="#profile_form_alert",
                    hx_swap="innerHTML",
                    hx_on_after_request="""
                        if(event.detail.successful && event.detail.xhr.responseText.includes('successfully')) {
                            setTimeout(function() {
                                document.getElementById('profile_modal_overlay').classList.remove('show');
                                document.getElementById('profile_form_container').classList.remove('show');
                                document.getElementById('profile_form').reset();
                                document.body.dispatchEvent(new CustomEvent('profileCreated'));
                            }, 1500);
                        }
                    """
                ),
                id="profile_form_container",
                cls="profile-form-container"
            )
        ),
        
        # Add CSS for the popup
        Style("""
            /* Modal Animation Classes */
            .profile-modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: rgba(0, 0, 0, 0.7);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 1000;
                opacity: 0;
                visibility: hidden;
                transition: opacity 0.3s ease, visibility 0.3s ease;
            }
            
            .profile-modal-overlay.show {
                opacity: 1;
                visibility: visible;
            }
            
            .profile-form-container {
                background: #111;
                padding: 24px;
                border-radius: 16px;
                width: 90%;
                max-width: 500px;
                position: fixed;
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%) scale(0.8);
                transition: all 0.3s cubic-bezier(0.68, -0.55, 0.27, 1.55);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                z-index: 1001;
                opacity: 0;
                visibility: hidden;
            }
            
            .profile-form-container.show {
                transform: translate(-50%, -50%) scale(1);
                opacity: 1;
                visibility: visible;
            }
            
            /* Form Input Styling */
            .profile-input {
                width: 100%;
                padding: 12px;
                margin-bottom: 16px;
                border-radius: 8px;
                background: #222;
                border: 1px solid #333;
                color: white;
                transition: border-color 0.3s ease;
            }
            
            .profile-input:focus {
                border-color: #f6cd70;
                outline: none;
            }
            
            /* Button Styling */
            .profile-btn {
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            
            .profile-btn.cancel-btn {
                background: #333;
                color: white;
            }
            
            .profile-btn.submit-btn {
                background: #f6cd70;
                color: black;
                font-weight: 800;
            }
            
            .profile-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }
            
            /* Response Message Styling */
            .profile-response {
                color: white;
                border-radius: 8px;
                margin-bottom: 16px;
            }
        """)
    )