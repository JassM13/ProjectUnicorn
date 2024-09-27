from fasthtml.common import *

app, rt = fast_app(hdrs=
                   Link(rel="script", href="https://kit.fontawesome.com/dd1d30eebf.js")
                   )

@rt("/")
def get():
    return Div(
        
        # Main container with sidebar and main content
        Div(
            # Rounded Sidebar container
            Div(
                # Logo at the top
                Div(
                    Img(src="./assets/logo.png", alt="ProjectUnicorn Logo", width="50px", height="50px", style="border-radius: 10%;"),
                    style="text-align:center; margin: 20px 0;"
                ),

                # Navigation icons in the middle
                Div(
                    A(I("", class_="fa-solid fa-house"), href="/", style="display:block; margin: 20px 0; text-align:center; font-size:24px; color:white;"),
                    A(I("", class_="fas fa-chart-line"), href="/analytics", style="display:block; margin: 20px 0; text-align:center; font-size:24px; color:white;"),
                    A(I("", class_="fas fa-cog"), href="/settings", style="display:block; margin: 20px 0; text-align:center; font-size:24px; color:white;"),
                ),
                
                # Settings icon at the bottom
                Div(
                    A(I("", class_="fas fa-user-cog"), href="/profile", style="display:block; margin-top: auto; text-align:center; font-size:24px; color:white;"),
                    style="position: absolute; bottom: 20px; width:100%;"
                ),
                
                # Sidebar Styling
                style="""
                    width: 70px; height: 95vh; background-color: #000; position: fixed; 
                    top: 20px; left: 20px; display: flex; flex-direction: column; 
                    justify-content: space-between; color:white; 
                    padding: 20px; border-radius: 16px; 
                """
            ),
            
            # Main content area on the right
            Div(
                H1("Hello, World!", style="text-align:center; margin-top: 100px;"),
                style="margin-left: 120px; padding: 20px; background-color: black; color: white;"
            )
        ),
        style="display:flex;"
    )

serve()
