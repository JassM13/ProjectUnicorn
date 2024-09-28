from fasthtml.common import *
from unicornManager import unicornAgent
import json

# Import the views
from views.components.sidebar import sidebar
from views.dashboardview import dashboard_view
from views.chatview import chat_view
from views.settings import settings_view

app, rt = fast_app(live=True,
                   hdrs=(picolink,
                     Style('@media only screen and (prefers-color-scheme:dark){:root:not([data-theme]){--pico-background-color:#f6cd70;'),
                     SortableJS('.sortable'))
                )

# Initialize the AI agent
agent = unicornAgent.AdvancedNLPAgent("BetaUser")


agent.interact("hey")

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
    print(request)
    data = await request.json()
    user_input = data.get("message", "")
    response = agent.process_input(user_input)
    return JsonResponse({"response": response})

# Default route redirects to dashboard
@rt("/")
def get_home():
    return Redirect("/dashboard")

# Serve the app
serve()