from fasthtml.common import *
from unicornManager import unicornAgent
import traceback
import asyncio

# Import the views
from views.components.sidebar import sidebar
from views.dashboardview import dashboard_view
from views.chatview import chat_view
from views.settings import settings_view

app, rt = fast_app(live=True,
                  hdrs=(picolink,
                    Style(""":root {--pico-spacing: 0rem;} @media only screen and (prefers-color-scheme:dark){:root:not([data-theme]){--pico-background-color:#f6cd70;"""),
                    SortableJS('.sortable'))
               )

# Initialize the AI agent
agent = unicornAgent.agent

# Route for dashboard
@rt("/dashboard")
def get_dashboard():
    return Div(
        sidebar(active="dashboard"),
        Div(dashboard_view(), style="margin-left: 100px;"),
        style="display:flex;"
    )

# Route for chat
@rt("/chat")
def get_chat():
    return Div(
        sidebar(active="chat"),
        Div(chat_view(), style="margin-left: 100px;"),
        style="display:flex;"
    )

# Route for settings
@rt("/settings")
def get_settings():
    return Div(
        sidebar(active="settings"),
        Div(settings_view(), style="margin-left: 100px;"),
        style="display:flex;"
    )

# Route to handle chat responses
@rt("/chat_response", methods=["POST"])
async def chat_response(request):
    try:
        data = await request.json()
        user_input = data.get("message", "")
        if not user_input.strip():
            return JSONResponse({"response": "Please enter a valid message."})
        
        # Process input and get response asynchronously
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, agent.generate_response, user_input)
        return JSONResponse({"response": response})
    except Exception as e:
        print(f"An error occurred in chat_response: {e}")
        print(traceback.format_exc())  # This will print the full traceback
        return JSONResponse({"response": "An error occurred processing your request."}, status_code=500)

# Default route redirects to dashboard
@rt("/")
def get_home():
    return Redirect("/dashboard")

# Serve the app
serve()