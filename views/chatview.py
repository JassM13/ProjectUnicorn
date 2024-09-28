from fasthtml.common import *
from unicornManager import unicornAgent

# Initialize the AI agent
agent = unicornAgent.AdvancedNLPAgent("BetaUser")

def chat_view():
    return Div(
        # Chat container
        Div(
            # Message display area (scrollable)
            Div(
                id="chat_box",
                style="""
                    flex-grow: 1; overflow-y: auto; padding: 10px; 
                    border-radius: 12px; background-color: #222; margin-bottom: 20px;
                    width: 100%; box-sizing: border-box;
                """
            ),
            style="flex-grow: 1; display: flex; flex-direction: column;"
        ),
        
        # Input section at the bottom
        Div(
            Input(
                type="text",
                placeholder="Type a message...",
                id="user_input",
                style="flex-grow: 1; padding: 10px; border-radius: 12px; border: none; margin-right: 10px; background-color: #333; height: 50px;"
            ),
            Button(
                "Send",
                id="send_button",
                style="padding: 10px 20px; background-color: #f6cd70; color: black; border: none; border-radius: 12px; cursor: pointer; height: 50px;"
            ),
            style="display: flex; width: 100%;"
        ),

        # Main chat area styling
        style="""
            display: flex; flex-direction: column; justify-content: space-between;
            padding: 20px; background-color: #000; color: white; 
            height: 95vh; border-radius: 16px; position: absolute; 
            right: 20px; top: 20px; left: 100px; bottom: 20px;
        """,
        script="""
            console.log('Script loaded');
            window.onload = function() {
                console.log('Window loaded');
                const sendButton = document.getElementById('send_button');
                const userInput = document.getElementById('user_input');
                const chatBox = document.getElementById('chat_box');

                sendButton.onclick = function() {
                    console.log('Send button clicked');
                    const inputText = userInput.value;
                    console.log('User input:', inputText);
                    if (inputText.trim() !== "") {
                        fetch('/chat_response', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ message: inputText })
                        })
                        .then(response => response.json())
                        .then(data => {
                            console.log('Response from server:', data);
                            const userMessage = document.createElement('p');
                            userMessage.textContent = 'User: ' + inputText;
                            userMessage.style.color = 'white';
                            userMessage.style.margin = '5px 0';
                            chatBox.appendChild(userMessage);

                            const aiMessage = document.createElement('p');
                            aiMessage.textContent = 'AI: ' + data.response;
                            aiMessage.style.color = 'white';
                            aiMessage.style.margin = '5px 0';
                            chatBox.appendChild(aiMessage);

                            userInput.value = ''; // Clear input
                            chatBox.scrollTop = chatBox.scrollHeight; // Scroll to bottom
                        })
                        .catch(error => console.error('Error:', error));
                    }
                };
            }
        """
    )
