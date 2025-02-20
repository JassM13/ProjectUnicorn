from fasthtml.common import *

def trades_view():
    return Div(
        Script("""
            document.addEventListener('DOMContentLoaded', function() {
                const addButton = document.getElementById('add_trade_button');
                const modal = document.getElementById('trade_modal');
                const modalOverlay = document.getElementById('modal_overlay');
                const cancelButton = document.getElementById('cancel_trade_button');

                function showModal() {
                    modalOverlay.style.opacity = '1';
                    modalOverlay.style.visibility = 'visible';
                    modal.style.opacity = '1';
                    modal.style.visibility = 'visible';
                    modal.style.transform = 'translate(-50%, -50%) scale(1)';
                }

                function hideModal() {
                    modalOverlay.style.opacity = '0';
                    modalOverlay.style.visibility = 'hidden';
                    modal.style.opacity = '0';
                    modal.style.visibility = 'hidden';
                    modal.style.transform = 'translate(-50%, -50%) scale(0.8)';
                }

                addButton.addEventListener('click', showModal);
                cancelButton.addEventListener('click', hideModal);
                modalOverlay.addEventListener('click', function(e) {
                    if (e.target === modalOverlay) hideModal();
                });
            });
        """),
        # Main container with trades list and form
        Div(
            # Left side - Trades List
            Div(
                H2("Your Trades", style="margin: 0 0 20px 0; color: #f6cd70;"),
                Div(
                    id="trades_list",
                    style="""
                        flex: 1;
                        overflow-y: auto;
                        padding: 10px;
                        display: flex;
                        flex-direction: column;
                        gap: 15px;
                        margin-bottom: 80px;
                    """
                ),
                id="trades_list_container",
                style="flex: 1; padding: 20px; background-color: #111; border-radius: 12px; position: relative; width: 100%; transition: width 0.3s ease;"
            ),
            
            # Modal Overlay
            Div(
                id="modal_overlay",
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
                """
            ),
            
            # Trade Form
            Div(
                H2("Add New Trade", style="margin: 0 0 20px 0; color: #f6cd70;"),
                Form(
                    Input(type="text", name="symbol", placeholder="Symbol", style="width: 100%; padding: 12px; margin-bottom: 16px; border-radius: 8px; background: #222; border: 1px solid #333; color: white;"),
                    Input(type="number", name="entry_price", placeholder="Entry Price", style="width: 100%; padding: 12px; margin-bottom: 16px; border-radius: 8px; background: #222; border: 1px solid #333; color: white;"),
                    Input(type="number", name="exit_price", placeholder="Exit Price", style="width: 100%; padding: 12px; margin-bottom: 16px; border-radius: 8px; background: #222; border: 1px solid #333; color: white;"),
                    Input(type="number", name="position_size", placeholder="Position Size", style="width: 100%; padding: 12px; margin-bottom: 16px; border-radius: 8px; background: #222; border: 1px solid #333; color: white;"),
                    Select(
                        Option("Long", value="long"),
                        Option("Short", value="short"),
                        name="trade_type",
                        style="width: 100%; padding: 12px; margin-bottom: 16px; border-radius: 8px; background: #222; border: 1px solid #333; color: white;"
                    ),
                    Textarea(name="notes", placeholder="Trade Notes", style="width: 100%; padding: 12px; margin-bottom: 16px; border-radius: 8px; background: #222; border: 1px solid #333; color: white; min-height: 100px;"),
                    Div(
                        Button(
                            "Cancel",
                            type="button",
                            id="cancel_trade_button",
                            style="padding: 12px 24px; background: #333; color: white; border: none; border-radius: 8px; cursor: pointer; margin-right: 10px;"
                        ),
                        Button(
                            "Add Trade",
                            type="submit",
                            style="padding: 12px 24px; background: #f6cd70; color: black; border: none; border-radius: 8px; cursor: pointer;"
                        ),
                        style="display: flex; justify-content: flex-end;"
                    ),
                    style="width: 100%;"
                ),
                id="trade_modal",
                style="""
                    background: #111;
                    padding: 32px;
                    border-radius: 16px;
                    width: 90%;
                    max-width: 600px;
                    max-height: 90vh;
                    position: fixed;
                    left: 50%;
                    top: 50%;
                    transform: translate(-50%, -50%) scale(0.8);
                    transition: all 0.3s ease;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    overflow-y: auto;
                    z-index: 1001;
                    opacity: 0;
                    visibility: hidden;
                """
            ),
            
            # Add Trade Button (fixed to bottom right)
            Button(
                "+ Add Trade",
                id="add_trade_button",
                style="""
                    position: fixed;
                    bottom: 50px;
                    right: 50px;
                    height: 50px;
                    width: auto;
                    padding: 0 24px;
                    border-radius: 16px;
                    background-color: #f6cd70;
                    color: black;
                    border: none;
                    font-size: 20px;
                    cursor: pointer;
                    z-index: 1000;
                """
            ),
            style="display: flex; height: 100%; position: relative;"
        ),
        style="""
            display: flex;
            flex-direction: column;
            padding: 20px;
            background-color: #000;
            color: white;
            height: 95vh;
            border-radius: 16px;
            position: absolute;
            right: 20px;
            top: 20px;
            left: 100px;
            bottom: 20px;
            overflow: hidden;
        """
    )