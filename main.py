from fasthtml.common import *
from fa6_icons import svgs,dims

app, rt = fast_app(live=true,
                   hdrs=(picolink,
                     # `Style` is an `FT` object, which are 3-element lists consisting of:
                     # (tag_name, children_list, attrs_dict).
                     # FastHTML composes them from trees and auto-converts them to HTML when needed.
                     # You can also use plain HTML strings in handlers and headers,
                     # which will be auto-escaped, unless you use `NotStr(...string...)`.
                     Style('@media only screen and (prefers-color-scheme:dark){:root:not([data-theme]){--pico-background-color:#f6cd70;'),
                     # Have a look at fasthtml/js.py to see how these Javascript libraries are added to FastHTML.
                     # They are only 5-10 lines of code each, and you can add your own too.
                     SortableJS('.sortable'))
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
                    A(Img(src="./assets/logo.png", alt="ProjectUnicorn Logo", width="50px", height="50px", style="border-radius: 10%;"), href="/"),
                    style="text-align:center; margin: 20px 0;"
                ),

                # Navigation icons in the middle
                Div(
                    A(I(svgs.house.solid), href="/", style="display:block; margin: 20px 0; text-align:center; font-size:16px; color:white; opacity: 1;"),
                    A(I(svgs.camera.solid), href="/analytics", style="display:block; margin: 20px 0; text-align:center; font-size:16px; color:white; opacity: 0.5;"),
                    A(I(svgs.gear.solid), href="/settings", style="display:block; margin: 20px 0; text-align:center; font-size:16px; color:white; opacity: 0.5;"),
                ),
                
                # Settings icon at the bottom
                Div(
                    A(I(svgs.user.solid), href="/profile", style="display:block; margin-top: 20px 0; text-align:center; font-size:16px; color:white; opacity: 1;"),
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
                H1("Hello, World!", style="text-align:center;"),
                style=""" padding: 20px; background-color: #000; 
                    color: white; height: 95vh; border-radius: 16px; 
                    display: flex; flex-direction: column; justify-content: center; 
                    align-items: center; position: absolute; right: 20px; top: 20px; 
                    left: 100px; bottom: 20px;
                """
            )

        ),
        style="display:flex;"
    )

serve()
