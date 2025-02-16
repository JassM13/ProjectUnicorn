from fasthtml.common import *
from views.components.sidebar import sidebar
from views.blankview import chat_view
import traceback
import asyncio

def register_chat_routes(rt):
    @rt("/chat")
    def get_chat():
        return Div(
            sidebar(active="chat"),
            Div(chat_view(), style="margin-left: 100px;"),
            style="display:flex;"
        )

    @rt("/chat_response", methods=["POST"])
    async def chat_response(request):
        try:
            data = await request.json()
            user_input = data.get("message", "")
            if not user_input.strip():
                return JSONResponse({"response": "Please enter a valid message."})
            
            # Process input and get response asynchronously
            loop = asyncio.get_event_loop()
        except Exception as e:
            print(f"An error occurred in chat_response: {e}")
            print(traceback.format_exc())  # This will print the full traceback
            return JSONResponse({"response": "An error occurred processing your request."}, status_code=500)

    return rt