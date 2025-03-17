from fasthtml.common import *

def accountview(session):
    return Div(
        Div(
            # Header Section
            Div(
                H2("Account Information", style="margin: 0 0 8px 0;"),
                style="margin-bottom: 24px;"
            ),
            
            # Account Details Section
            Div(
                # Profile Section
                Div(
                    H3("Profile", style="color: #f6cd70; margin: 0 0 16px 0;"),
                    Div(
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
                        Div(
                            Label("User ID", style="color: #666; font-size: 14px;"),
                            P(session.get('user', {}).get('user_id', 'Not available'),
                              style="margin: 4px 0 16px 0; font-size: 16px;")
                        ),
                        style="background: #1a1a1a; padding: 20px; border-radius: 12px; margin-bottom: 24px;"
                    )
                ),
                
                # Security Section
                Div(
                    H3("Security", style="color: #f6cd70; margin: 0 0 16px 0;"),
                    Div(
                        Button(
                            "Change Password",
                            style="""background-color: #333; color: white; border: none;
                                   border-radius: 8px; padding: 8px 16px; font-size: 14px;
                                   cursor: pointer; transition: background-color 0.2s;""",
                            hx_on_mouseenter="this.style.backgroundColor='#444'",
                            hx_on_mouseleave="this.style.backgroundColor='#333'"
                        ),
                        style="background: #1a1a1a; padding: 20px; border-radius: 12px;"
                    )
                ),
                style="width: 100%; max-width: 600px;"
            ),
            style="width: 100%;"
        ),
        style="""
            display: flex; flex-direction: column; padding: 30px;
            color: white; height: 95vh; border-radius: 16px;
            position: absolute; right: 20px; top: 20px; left: 100px;
            bottom: 20px; overflow: auto; border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        """
    )