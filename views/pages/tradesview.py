from fasthtml.common import *

def trades_view():
    return Div(
        Script("""
            document.addEventListener('htmx:load', function() {
                // Add a small delay to ensure DOM is fully loaded
                setTimeout(function() {
                    const addButton = document.getElementById('add_trade_button');
                    const modal = document.getElementById('trade_modal');
                    const modalOverlay = document.getElementById('modal_overlay');
                    const cancelButton = document.getElementById('cancel_trade_button');
                    const tradeForm = document.getElementById('trade_form');

                    function showModal() {
                        if (!modalOverlay || !modal) return;
                        modalOverlay.style.opacity = '1';
                        modalOverlay.style.visibility = 'visible';
                        modal.style.opacity = '1';
                        modal.style.visibility = 'visible';
                        modal.style.transform = 'translate(-50%, -50%) scale(1)';
                    }

                    function hideModal() {
                        if (!modalOverlay || !modal) return;
                        modalOverlay.style.opacity = '0';
                        modalOverlay.style.visibility = 'hidden';
                        modal.style.opacity = '0';
                        modal.style.visibility = 'hidden';
                        modal.style.transform = 'translate(-50%, -50%) scale(0.8)';
                        if (tradeForm) tradeForm.reset();
                    }

                    if (addButton) addButton.addEventListener('click', showModal);
                    if (cancelButton) cancelButton.addEventListener('click', hideModal);
                    if (modalOverlay) modalOverlay.addEventListener('click', function(e) {
                        if (e.target === modalOverlay) hideModal();
                    });
                }, 100); // Small delay to ensure DOM elements are available
            });
        """),
        # Main container with trades list and form
        Div(
            # Left side - Trades List
            Div(
                H2("Trades", style="margin: 0;"),
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
                    Select(
                        Option("Futures", value="futures", selected=True),
                        Option("Options", value="options"),
                        Option("Stocks", value="stocks"),
                        name="instrument_type",
                        required=True,
                        style="width: 100%; padding: 12px; margin-bottom: clamp(8px, 1.5vh, 16px); border-radius: 8px; background: #222; border: 1px solid #333; color: white;"
                    ),
                    Input(type="text", name="symbol", placeholder="Contract Name", required=True, style="width: 100%; padding: clamp(8px, 1.5vh, 12px); margin-bottom: clamp(8px, 1.5vh, 16px); border-radius: 8px; background: #222; border: 1px solid #333; color: white;"),
                    Div(
                        Div(
                            Input(type="datetime-local", name="entered_at", required=True, style="width: 100%; padding: clamp(8px, 1.5vh, 12px); border-radius: 8px; background: #222; border: 1px solid #333; color: white;"),
                            style="flex: 1; margin-right: 8px;"
                        ),
                        Div(
                            Input(type="datetime-local", name="exited_at", required=True, style="width: 100%; padding: clamp(8px, 1.5vh, 12px); border-radius: 8px; background: #222; border: 1px solid #333; color: white;"),
                            style="flex: 1;"
                        ),
                        style="display: flex; margin-bottom: clamp(8px, 1.5vh, 16px);"
                    ),
                    Input(type="hidden", name="trade_day", required=True),
                    Select(
                        Option("Long", value="long"),
                        Option("Short", value="short"),
                        name="type",
                        required=True,
                        style="width: 100%; padding: clamp(8px, 1.5vh, 12px); margin-bottom: clamp(8px, 1.5vh, 16px); border-radius: 8px; background: #222; border: 1px solid #333; color: white;"
                    ),
                    Div(
                        Div(
                            Input(type="number", name="entry_price", placeholder="Entry Price", step="0.01", style="width: 100%; padding: clamp(8px, 1.5vh, 12px); border-radius: 8px; background: #222; border: 1px solid #333; color: white;"),
                            style="flex: 1; margin-right: 8px;"
                        ),
                        Div(
                            Input(type="number", name="exit_price", placeholder="Exit Price", step="0.01", style="width: 100%; padding: clamp(8px, 1.5vh, 12px); border-radius: 8px; background: #222; border: 1px solid #333; color: white;"),
                            style="flex: 1;"
                        ),
                        style="display: flex; margin-bottom: clamp(8px, 1.5vh, 16px);"
                    ),
                    Input(type="number", name="size", placeholder="Position Size", step="0.01", style="width: 100%; padding: clamp(8px, 1.5vh, 12px); margin-bottom: clamp(8px, 1.5vh, 16px); border-radius: 8px; background: #222; border: 1px solid #333; color: white;"),
                    Textarea(name="notes", placeholder="Trade Notes", style="width: 100%; padding: clamp(8px, 1.5vh, 12px); margin-bottom: clamp(8px, 1.5vh, 16px); border-radius: 8px; background: #222; border: 1px solid #333; color: white; min-height: clamp(60px, 10vh, 100px);"),
                    Div(
                        id="trade_form_alert",
                        style="""
                            color: white;
                            border-radius: 8px;
                            padding: 12px;
                        """
                    ),
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
                    Script("""
                        document.addEventListener('htmx:load', function() {
                            setTimeout(function() {
                                const now = new Date();
                                const enteredAtInput = document.querySelector('input[name="entered_at"]');
                                const exitedAtInput = document.querySelector('input[name="exited_at"]');
                                const tradeDayInput = document.querySelector('input[name="trade_day"]');
                                
                                function formatDateTime(date) {
                                    return date.toISOString().slice(0, 16);
                                }
                                
                                // Set initial values with null checks
                                if (enteredAtInput) enteredAtInput.value = formatDateTime(now);
                                if (exitedAtInput) exitedAtInput.value = formatDateTime(now);
                                if (tradeDayInput) tradeDayInput.value = now.toISOString().split('T')[0];
                                
                                // Update trade_day when exited_at changes
                                if (exitedAtInput && tradeDayInput) {
                                    exitedAtInput.addEventListener('change', function() {
                                        const exitDate = new Date(this.value);
                                        tradeDayInput.value = exitDate.toISOString().split('T')[0];
                                    });
                                }
                            }, 100); // Small delay to ensure elements are available
                        });
                    """),
                    Div(
                        id="trade_form_error",
                    ),
                    id="trade_form",
                    style="width: 100%;",
                    hx_post="/api/trades",
                    hx_target="#trade_form_alert",
                    hx_swap="innerHTML"
                ),
                id="trade_modal",
                style="""
                    background: #111;
                    padding: clamp(16px, 3vh, 32px);
                    border-radius: 16px;
                    width: 90%;
                    max-width: 600px;
                    height: auto;
                    max-height: 85vh;
                    position: fixed;
                    left: 50%;
                    top: 50%;
                    transform: translate(-50%, -50%) scale(0.8);
                    transition: all 0.3s ease;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    z-index: 1001;
                    opacity: 0;
                    visibility: hidden;
                    display: flex;
                    flex-direction: column;
                """
            ),
            
            # Add Trade Button (fixed to bottom right)
            Button(
                "+ Add Trade",
                id="add_trade_button",
                style="""
                    position: fixed;
                    bottom: 40px;
                    right: 40px;
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
            padding: 30px;
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