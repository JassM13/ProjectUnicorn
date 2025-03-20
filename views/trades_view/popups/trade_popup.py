from fasthtml.common import *

def trade_popup():
    """
    Creates a popup for adding new trades.
    This implementation uses HTMX for animations and interactions.
    """
    return Div(
        # Overlay with HTMX animation classes
        Div(
            id="trade_modal_overlay",
            cls="trade-modal-overlay",
            hx_on_click="this.classList.remove('show'); document.getElementById('trade_form_container').classList.remove('show')",
        ),
        
        # Trade Form Container
        Div(
            # Form Content
            Div(
                H2("Add New Trade", style="margin: 0 0 20px 0; color: #f6cd70;"),
                
                Form(
                    # Two-column layout container
                    Div(
                        # Left column - Trade details
                        Div(
                            # Trade Type Selection
                            Div(
                                Label("Trade Type", style="color: white; display: block; margin-bottom: 8px;"),
                                Div(
                                    Button("Buy", 
                                        id="buy_btn",
                                        type="button", 
                                        cls="trade-type-btn selected",
                                        hx_on_click="htmx.toggleClass(this, 'selected'); htmx.toggleClass(document.getElementById('sell_btn'), 'selected'); document.getElementById('trade_type').value='buy'",
                                        style="border-radius: 8px 0 0 8px;"
                                    ),
                                    Button("Sell", 
                                        id="sell_btn",
                                        type="button", 
                                        cls="trade-type-btn",
                                        hx_on_click="htmx.toggleClass(this, 'selected'); htmx.toggleClass(document.getElementById('buy_btn'), 'selected'); document.getElementById('trade_type').value='sell'",
                                        style="border-radius: 0 8px 8px 0;"
                                    ),
                                    Input(type="hidden", name="trade_type", id="trade_type", value="buy"),
                                    style="display: flex; margin-bottom: 16px;"
                                ),
                            ),
                            
                            # Instrument Type Selection - Dropdown
                            Div(
                                Label("Instrument Type", style="color: white; display: block; margin-bottom: 8px;"),
                                Select(
                                    Option("Stocks", value="stocks", selected=True),
                                    Option("Forex", value="forex"),
                                    Option("Futures", value="futures"),
                                    Option("Crypto", value="crypto"),
                                    name="instrument_type",
                                    id="instrument_type",
                                    cls="trade-input",
                                    style="margin-bottom: 16px;"
                                ),
                            ),
                            
                            # Symbol Input
                            Div(
                                Label("Symbol", style="color: white; display: block; margin-bottom: 8px;"),
                                Input(
                                    type="text", 
                                    name="symbol", 
                                    placeholder="e.g. AAPL", 
                                    required=True,
                                    cls="trade-input"
                                ),
                                style="margin-bottom: 16px;"
                            ),
                            
                            # Price and Quantity
                            Div(
                                Div(
                                    Label("Price ($)", style="color: white; display: block; margin-bottom: 8px;"),
                                    Input(
                                        type="number", 
                                        name="price", 
                                        step="0.01",
                                        min="0.01",
                                        placeholder="0.00", 
                                        required=True,
                                        cls="trade-input"
                                    ),
                                    style="flex: 1; margin-right: 12px;"
                                ),
                                Div(
                                    Label("Quantity", style="color: white; display: block; margin-bottom: 8px;"),
                                    Input(
                                        type="number", 
                                        name="quantity",
                                        min="1",
                                        step="1", 
                                        placeholder="0", 
                                        required=True,
                                        cls="trade-input"
                                    ),
                                    style="flex: 1;"
                                ),
                                style="display: flex; margin-bottom: 16px;"
                            ),
                            
                            # Date and Time
                            Div(
                                Label("Date & Time", style="color: white; display: block; margin-bottom: 8px;"),
                                Input(
                                    type="datetime-local", 
                                    name="datetime", 
                                    required=True,
                                    cls="trade-input"
                                ),
                                style="margin-bottom: 16px;"
                            ),
                            style="flex: 1; margin-right: 16px;"
                        ),
                        
                        # Right column - Notes
                        Div(
                            Label("Notes", style="color: white; display: block; margin-bottom: 8px;"),
                            Textarea(
                                name="notes",
                                placeholder="Add any trade notes here...",
                                cls="trade-input",
                                style="min-height: 100%; resize: vertical; resize: none;"
                            ),
                            style="flex: 1; display: flex; flex-direction: column;"
                        ),
                        style="display: flex; margin-bottom: 20px;"
                    ),
                    
                    # Response message area
                    Div(
                        id="trade_form_response",
                        cls="trade-response"
                    ),
                    
                    # Action Buttons
                    Div(
                        Button("Cancel", 
                            type="button",
                            cls="trade-btn cancel-btn",
                            hx_on_click="document.getElementById('trade_modal_overlay').classList.remove('show'); document.getElementById('trade_form_container').classList.remove('show'); this.form.reset();"
                        ),
                        Button("Add Trade", 
                            type="submit",
                            cls="trade-btn submit-btn"
                        ),
                        style="display: flex; justify-content: flex-end; gap: 12px;"
                    ),
                    
                    id="trade_form",
                    hx_post="/api/trades",
                    hx_target="#trade_form_response",
                    hx_swap="innerHTML",
                    hx_on_after_request="""
                        if(event.detail.successful) {
                            setTimeout(function() {
                                document.getElementById('trade_modal_overlay').classList.remove('show');
                                document.getElementById('trade_form_container').classList.remove('show');
                                document.getElementById('trade_form').reset();
                                htmx.trigger('#tradesGrid', 'htmx:refresh');
                            }, 1500);
                        }
                    """
                )
            ),
            id="trade_form_container",
            cls="trade-form-container",
            
        ),
        
        # Add CSS for the popup
        Style("""
            /* Modal Animation Classes */
            .trade-modal-overlay {
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
            }
            
            .trade-modal-overlay.show {
                opacity: 1;
                visibility: visible;
            }
            
            .trade-form-container {
                background: #111;
                padding: 24px;
                border-radius: 16px;
                width: 90%;
                max-width: 80%;
                position: fixed;
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%) scale(0.8);
                transition: all 0.3s cubic-bezier(0.68, -0.55, 0.27, 1.55);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                z-index: 1001;
                opacity: 0;
                visibility: hidden;
            }
            
            .trade-form-container.show {
                transform: translate(-50%, -50%) scale(1);
                opacity: 1;
                visibility: visible;
            }
            
            /* Form Styling */
            .trade-input {
                width: 100%;
                padding: 12px;
                border-radius: 8px;
                background: #222;
                border: 1px solid #333;
                color: white;
                transition: border-color 0.3s ease, box-shadow 0.3s ease;
            }
            
            .trade-input:focus {
                border-color: #f6cd70;
                box-shadow: 0 0 0 2px rgba(246, 205, 112, 0.3);
                outline: none;
            }
            
            .trade-type-btn {
                flex: 1;
                padding: 10px;
                background: #222;
                color: white;
                border: 1px solid #333;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .trade-type-btn:hover:not(.selected) {
                background: #333;
            }
            
            .trade-type-btn.selected {
                background: #f6cd70;
                color: black;
                font-weight: bold;
            }
            
            .trade-btn {
                padding: 12px 24px;
                border-radius: 8px;
                border: none;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            
            .cancel-btn {
                background: #333;
                color: white;
            }
            
            .cancel-btn:hover {
                background: #444;
            }
            
            .submit-btn {
                background: #f6cd70;
                color: black;
            }
            
            .submit-btn:hover {
                background: #e5bc61;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            
            /* Response Message Styling */
            .trade-response {
                margin-bottom: 16px;
                padding: 12px;
                border-radius: 8px;
                transition: all 0.3s ease;
            }
            
            .trade-response.success {
                background-color: rgba(51, 204, 51, 0.2);
                border: 1px solid #33cc33;
                color: #33cc33;
            }
            
            .trade-response.error {
                background-color: rgba(255, 68, 68, 0.2);
                border: 1px solid #ff4444;
                color: #ff4444;
            }
            
            /* Animation for form elements */
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .trade-form-container.show form > div {
                animation: fadeInUp 0.4s ease forwards;
                opacity: 0;
            }
            
            .trade-form-container.show form > div:nth-child(1) { animation-delay: 0.1s; }
            .trade-form-container.show form > div:nth-child(2) { animation-delay: 0.15s; }
            .trade-form-container.show form > div:nth-child(3) { animation-delay: 0.2s; }
            .trade-form-container.show form > div:nth-child(4) { animation-delay: 0.25s; }
            .trade-form-container.show form > div:nth-child(5) { animation-delay: 0.3s; }
            .trade-form-container.show form > div:nth-child(6) { animation-delay: 0.35s; }
            .trade-form-container.show form > div:nth-child(7) { animation-delay: 0.4s; }
        """)
    )