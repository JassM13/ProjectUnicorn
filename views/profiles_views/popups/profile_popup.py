from fasthtml.common import *

def profile_popup():
    return Div(
        # Modal overlay with HTMX attributes for showing/hiding
        Div(
            id="profile_modal_overlay",
            hx_swap_oob="true",
            style="""
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
            """,
            # Close modal when clicking on overlay
            hx_on__click="""if(event.target === this) {
                this.style.opacity = '0';
                this.style.visibility = 'hidden';
                document.querySelector('#modal_content').style.opacity = '0';
                document.querySelector('#modal_content').style.visibility = 'hidden';
                document.querySelector('#modal_content').style.transform = 'translate(-50%, -50%) scale(0.8)';
            }"""
        ),
        
        # Profile Form Container
        Div(
            Div(
                H2("Create New Profile", style="margin: 0 0 20px 0; color: #f6cd70;"),
                Form(
                    Input(type="text", name="profile_name", placeholder="Profile Name", required=True, 
                         style="width: 100%; padding: 12px; margin-bottom: 16px; border-radius: 8px; background: #222; border: 1px solid #333; color: white;"),
                    
                    Div(
                        Label(
                            Input(type="checkbox", name="broker_account", style="margin-right: 8px;", disabled=True),
                            Span("Connect to Broker Account ", 
                                 Span("(unavailable)", style="color: #ff6b6b; font-size: 12px; font-style: italic;")),
                            style="display: flex; align-items: center; color: white;"
                        ),
                        style="margin-bottom: 16px;"
                    ),
        
                    Div(
                        id="profile_form_alert",
                        style="""
                            color: white;
                            border-radius: 8px;
                            margin-bottom: 16px;
                        """
                    ),
                    
                    Div(
                        Button(
                            "Cancel",
                            type="button",
                            style="padding: 12px 24px; background: #333; color: white; border: none; border-radius: 8px; cursor: pointer; margin-right: 10px;",
                            # Close modal with HTMX
                            onclick="closeProfileModal()"
                        ),
                        Button(
                            "Create Profile",
                            type="submit",
                            style="padding: 12px 24px; background: #f6cd70; color: black; border: none; border-radius: 8px; cursor: pointer; font-weight: 800;"
                        ),
                        style="display: flex; justify-content: flex-end;"
                    ),
                    
                    id="profile_form",
                    style="width: 100%;",
                    hx_post="/api/profiles",
                    hx_target="#profile_form_alert",
                    hx_swap="innerHTML",
                    # After successful submission, trigger profile refresh and close modal
                    hx_on__htmx_after_request="""
                        if(event.detail.successful && event.detail.xhr.responseText.includes('successfully')) {
                            setTimeout(function() {
                                // Hide modal
                                document.getElementById('profile_modal_overlay').style.opacity = '0';
                                document.getElementById('profile_modal_overlay').style.visibility = 'hidden';
                                document.querySelector('#modal_content').style.opacity = '0';
                                document.querySelector('#modal_content').style.visibility = 'hidden';
                                document.querySelector('#modal_content').style.transform = 'translate(-50%, -50%) scale(0.8)';
                                // Reset form
                                document.getElementById('profile_form').reset();
                                // Trigger refresh event
                                document.body.dispatchEvent(new CustomEvent('profileCreated'));
                            }, 1500);
                        }
                    """
                ),
                id="modal_content",
                style="""
                    background: #111;
                    padding: 24px;
                    border-radius: 16px;
                    width: 90%;
                    max-width: 500px;
                    position: fixed;
                    left: 50%;
                    top: 50%;
                    transform: translate(-50%, -50%) scale(0.8);
                    transition: all 0.3s ease;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    z-index: 1001;
                    opacity: 0;
                    visibility: hidden;
                """
            ),
            id="profile_popup_container"
        )
    )