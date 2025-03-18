from fasthtml.common import *

def subscription_popup(session):
    """
    Creates a popup for managing subscription tiers.
    This implementation uses HTMX for animations and interactions.
    """
    return Div(
        # Overlay with HTMX animation classes
        Div(
            id="subscription_modal_overlay",
            cls="subscription-modal-overlay",
            hx_on_click="this.classList.remove('show'); document.getElementById('subscription_container').classList.remove('show')",
        ),
        
        # Subscription Container
        Div(
            # Content
            Div(
                H2("Choose Your Plan", style="margin: 0 0 20px 0; color: #f6cd70; text-align: center;"),
                P("Select the plan that best fits your needs", 
                  style="text-align: center; color: #888; margin-bottom: 30px;"),
                
                # Subscription Cards Container
                Div(
                    # Free Tier
                    Div(
                        H3("Free", style="color: #4CAF50; margin: 0 0 16px 0;"),
                        P("$0", style="font-size: 32px; margin: 0 0 0 0;"),
                        P("/month", style="color: #888; margin: 0 0 24px 0;"),
                        Ul(
                            Li("Locked to 2 Profiles"),
                            Li("Limited Trade History"),
                            Li("No Image Uploads"),
                            Li("No Market News"),
                            style="list-style: none; padding: 0; margin: 0 0 24px 0; color: #ddd; overflow: hidden;"
                        ),
                        Div(
                            Button("Current Plan",
                                style="width: 90%; background-color: #262626; color: white; border: none; border-radius: 8px; padding: 8px 16px; font-size: 14px; cursor: pointer; margin-top: 12px; transition: all 0.2s ease;",
                                disabled=True,
                                id="pro_btn"
                            ) if session.get('user', {}).get('subscription_tier') == 'Free' else
                            Button("Downgrade to Free",  
                                style="width: 90%; background-color: #f6cd70; color: black; border: none; border-radius: 8px; padding: 8px 16px; font-size: 14px; cursor: pointer; margin-top: 12px; transition: all 0.2s ease;",
                                hx_post="/api/subscription/change",
                                hx_vals='{"plan": "free"}',
                                hx_target="#subscription_response"
                            ),
                            style="bottom: 24px; left: 24px; right: 24px;"
                        ),
                        style="text-align: center;"
                    ),
                    
                    # Pro Tier
                    Div(
                        H3("Pro", style="color: #2196F3; margin: 0 0 16px 0;"),
                        P("$10", style="font-size: 32px; margin: 0 0 0 0;"),
                        P("/month", style="color: #888; margin: 0 0 24px 0;"),
                        Ul(
                            Li("Up to 5 Profiles"),
                            Li("Full Trade History Access"),
                            Li("Image Uploads Enabled"),
                            Li("Market News"),
                            style="list-style: none; padding: 0; margin: 0 0 24px 0; color: #ddd; overflow: hidden;"
                        ),
                        Div(
                            Button("Current Plan",
                                style="width: 90%; background-color: #262626; color: white; border: none; border-radius: 8px; padding: 8px 16px; font-size: 14px; cursor: pointer; margin-top: 12px; transition: all 0.2s ease;",
                                disabled=True,
                                id="pro_btn"
                            ) if session.get('user', {}).get('subscription_tier') == 'Pro' else
                            Button(session.get('user', {}).get('subscription_tier') == 'Premium' and "Downgrade to Pro" or "Upgrade to Pro",      
                                style="width: 90%; background-color: #f6cd70; color: black; border: none; border-radius: 8px; padding: 8px 16px; font-size: 14px; cursor: pointer; margin-top: 12px; transition: all 0.2s ease;",
                                hx_post="/api/subscription/change",
                                hx_vals='{"plan": "pro"}',
                                hx_target="#subscription_response"
                            ),
                            style="bottom: 24px; left: 24px; right: 24px;"
                        ),
                        style="text-align: center;"
                    ),
                    
                    # Premium Tier
                    Div(
                        H3("Premium", style="color: #F6CD70; margin: 0 0 16px 0;"),
                        P("$30", style="font-size: 32px; margin: 0 0 0 0;"),
                        P("/month", style="color: #888; margin: 0 0 24px 0;"),
                        Ul(
                            Li("Everything in Pro"),
                            Li("Unlimited Profiles"),
                            Li("Advanced Analytics"),
                            Li("Priority Support"),
                            style="list-style: none; padding: 0; margin: 0 0 24px 0; color: #ddd; overflow: hidden;"
                        ),
                        Button("Current Plan",
                            style="width: 90%; background-color: #262626; color: white; border: none; border-radius: 8px; padding: 8px 16px; font-size: 14px; cursor: pointer; margin-top: 12px; transition: all 0.2s ease;",
                            disabled=True,
                            id="premium_btn"
                        ) if session.get('user', {}).get('subscription_tier') == 'Premium' else
                        Button("Upgrade to Premium",
                            style="width: 90%; background-color: #f6cd70; color: black; border: none; border-radius: 8px; padding: 8px 16px; font-size: 14px; cursor: pointer; margin-top: 12px; transition: all 0.2s ease;",
                            hx_post="/api/subscription/change",
                            hx_vals='{"plan": "premium"}',
                            hx_target="#subscription_response"
                        ),
                        style="text-align: center;"
                    ),
                    style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 0 auto;"
                ),
                
                # Close button
                Button("Close", 
                    type="button",
                    cls="close-btn",
                    hx_on_click="document.getElementById('subscription_modal_overlay').classList.remove('show'); document.getElementById('subscription_container').classList.remove('show');",
                    style="position: absolute; top: 16px; left: 16px;"
                ),
            ),
            id="subscription_container",
            cls="subscription-container",
        ),
        
        # Add CSS for the popup
        Style("""
            .subscription-modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: rgba(0, 0, 0, 0.7);
                backdrop-filter: blur(4px);
                -webkit-backdrop-filter: blur(4px);
                z-index: 1000;
                opacity: 0;
                visibility: hidden;
                transition: opacity 0.3s ease, visibility 0.3s ease;
            }
            
            .subscription-modal-overlay.show {
                opacity: 1;
                visibility: visible;
            }
            
            .subscription-container {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%) scale(0.95);
                background: rgba(26, 26, 26, 0.95);
                padding: 40px;
                border-radius: 16px;
                width: 90%;
                max-width: 1000px;
                max-height: 90vh;
                overflow-y: auto;
                z-index: 1001;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                opacity: 0;
                visibility: hidden;
                transition: all 0.3s cubic-bezier(0.68, -0.55, 0.27, 1.55);
            }
            
            .subscription-container.show {
                transform: translate(-50%, -50%) scale(1);
                opacity: 1;
                visibility: visible;
            }
            
            .close-btn {
                padding: 12px 32px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                background-color: #333;
                color: white;
            }
            
            .close-btn:hover {
                background-color: #444;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }
            
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            @keyframes slideIn {
                from { 
                    opacity: 0;
                    transform: translate(-50%, -48%);
                }
                to { 
                    opacity: 1;
                    transform: translate(-50%, -50%);
                }
            }
        """)
    )