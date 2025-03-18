from fasthtml.common import *
from views.account_views.popups.subscription_popup import subscription_popup

def accountview(session):
    # Helper function to get subscription tier details
    def get_tier_details(tier):
        tier_styles = {
            'Free': {'color': '#4CAF50'},
            'Premium': {'color': '#2196F3'},
            'Enterprise': {'color': '#F6CD70'}
        }
        return tier_styles.get(tier, tier_styles['Free'])
    
    # Get user's subscription tier
    user_tier = session.get('user', {}).get('subscription_tier', 'Free')
    tier_details = get_tier_details(user_tier)
    
    return Div(
        Div(
            # Header Section
            Div(
                H2("Account Information", style="margin: 0 0 8px 0;"),
                style="margin-bottom: 24px;"
            ),
            
            # Account Details Section
            Div(
                # Grid Container for all sections
                Div(
                    # Profile Section
                    Div(
                        Div(
                            H3("Profile", style="color: #f6cd70; margin: 0 0 16px 0;"),
                            Div(
                                Label("Username", style="color: #666; font-size: 14px;"),
                                P(session.get('user', {}).get('username', 'Not available'),
                                  style="margin: 4px 0 16px 0; font-size: 16px;")
                            ),
                            Div(
                                Label("Email", style="color: #666; font-size: 14px;"),
                                P(session.get('user', {}).get('email', 'Not available'),
                                  style="margin: 4px 0 16px 0; font-size: 16px;")
                            ),
                            style="""background: rgba(26, 26, 26, 0.8); padding: 24px; border-radius: 16px;
                                   backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
                                   border: 1px solid rgba(255, 255, 255, 0.1);
                                   box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);"""
                        )
                    ),

                    # Subscription Tier Section
                    Div(
                        Div(
                            H3("Subscription", style="color: #f6cd70; margin: 0 0 16px 0;"),
                            Div(
                                Span(f"{user_tier} Tier",
                                     style=f"color: {tier_details['color']}; font-size: 18px; font-weight: bold;"),
                                style="margin-bottom: 12px;"
                            ),
                            
                            Div(
                                P("Next billing date: " + session.get('user', {}).get('next_billing_date', 'N/A'),
                                  style="color: #666; margin: 4px 0;")
                            ) if user_tier != 'Free' else None,
                            Button(
                                "Upgrade Plan",
                                id="upgrade_plan_button",
                                style="""background-color: #f6cd70; color: black; border: none;
                                       border-radius: 8px; padding: 8px 16px; font-size: 14px;
                                       cursor: pointer; margin-top: 12px; transition: all 0.2s ease;
                                       hover:opacity: 0.9;""",
                                hx_on_click="document.getElementById('subscription_modal_overlay').classList.add('show'); document.getElementById('subscription_container').classList.add('show');"),
                            style="""background: rgba(26, 26, 26, 0.8); padding: 24px; border-radius: 16px;
                                   backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
                                   border: 1px solid rgba(255, 255, 255, 0.1);
                                   box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);"""
                        )
                    ),

                    # Account Statistics Section
                    Div(
                        Div(
                            H3("Account Statistics", style="color: #f6cd70; margin: 0 0 16px 0;"),
                            Div(
                                Label("Member Since", style="color: #666; font-size: 14px;"),
                                P(session.get('user', {}).get('created_at', 'Not available'),
                                  style="margin: 4px 0 16px 0; font-size: 16px;")
                            ),
                            Div(
                                Label("Last Login", style="color: #666; font-size: 14px;"),
                                P(session.get('user', {}).get('last_login', 'Not available'),
                                  style="margin: 4px 0 16px 0; font-size: 16px;")
                            ),
                            style="""background: rgba(26, 26, 26, 0.8); padding: 24px; border-radius: 16px;
                                   backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
                                   border: 1px solid rgba(255, 255, 255, 0.1);
                                   box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);"""
                        )
                    ),
                    
                    # Security Section
                    Div(
                        Div(
                            H3("Security", style="color: #f6cd70; margin: 0 0 16px 0;"),
                            Div(
                                Button(
                                    "Change Password",
                                    style="""background-color: #333; color: white; border: none;
                                           border-radius: 8px; padding: 8px 16px; font-size: 14px;
                                           cursor: pointer; transition: all 0.2s ease;""",
                                    hx_on_mouseenter="this.style.backgroundColor='#444'",
                                    hx_on_mouseleave="this.style.backgroundColor='#333'"
                                ),
                                Button(
                                    "Enable 2FA",
                                    style="""background-color: #333; color: white; border: none;
                                           border-radius: 8px; padding: 8px 16px; font-size: 14px;
                                           cursor: pointer; transition: all 0.2s ease;
                                           margin-left: 8px;""",
                                    hx_on_mouseenter="this.style.backgroundColor='#444'",
                                    hx_on_mouseleave="this.style.backgroundColor='#333'"
                                ),
                            ),
                            style="""background: rgba(26, 26, 26, 0.8); padding: 24px; border-radius: 16px;
                                   backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
                                   border: 1px solid rgba(255, 255, 255, 0.1);
                                   box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);"""
                        )
                    ),
                    style="""display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                           gap: 24px; width: 100%; height: 100%;"""
                ),
                style="width: 100%; height: 100%; display: flex; flex-direction: column;"
            ),
            style="width: 100%; height: 100%; display: flex; flex-direction: column;"
        ),
        # Add subscription popup container
        subscription_popup(session),
        style="""
            display: flex; flex-direction: column; padding: 30px;
            color: white; height: 95vh; border-radius: 16px;
            position: absolute; right: 20px; top: 20px; left: 100px;
            bottom: 20px; overflow: auto; border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        """
    )