from fasthtml.common import *

def create_grid_table(data):
    # Create table headers
    headers = ['Name', 'Trades', 'Last Updated', 'Broker Account', 'Actions']
    
    return Table(
        Thead(
            Tr(
                *[Th(header, 
                    hx_get=f"/api/profiles/get?sort={header.lower()}",
                    hx_target="#profilesGrid",
                    hx_swap="innerHTML",
                    style="background: #121212; color: #ffffff; padding: 16px; text-align: left; font-weight: 600; cursor: pointer; user-select: none;"
                ) for header in headers]
            )
        ),
        Tbody(
            cls="sortable",
            *[Tr(
                Td(item.get('name', ''), style=f"padding: 4px 0 4px 16px; {'border-bottom: 1px solid #2a2a2a;' if idx < len(data or []) - 1 else ''} background-color: #000; color: #fff;"),
                Td(str(item.get('trades', 0)), style=f"padding: 4px 0 4px 16px; {'border-bottom: 1px solid #2a2a2a;' if idx < len(data or []) - 1 else ''} background-color: #000; color: #fff;"),
                Td(item.get('last_updated', ''), style=f"padding: 4px 0 4px 16px; {'border-bottom: 1px solid #2a2a2a;' if idx < len(data or []) - 1 else ''} background-color: #000; color: #fff;"),
                Td('True' if item.get('broker_account') else 'False', style=f"padding: 4px 0 4px 16px; {'border-bottom: 1px solid #2a2a2a;' if idx < len(data or []) - 1 else ''} background-color: #000; color: #fff;"),
                Td(
                    Button(
                        Img(src='/assets/svgs/Edit/Edit_Pencil.svg', alt='Edit'),
                        style="background: none; border: none; cursor: pointer; padding: 4px; filter: brightness(0) saturate(100%) invert(91%) sepia(9%) saturate(2661%) hue-rotate(335deg) brightness(60%) contrast(80%);",
                        hx_get=f"/api/profiles/{item.get('id')}/edit",
                        hx_target="#profile_form",
                        onmouseover="this.style.filter='brightness(0) saturate(100%) invert(91%) sepia(9%) saturate(2661%) hue-rotate(335deg) brightness(99%) contrast(80%)'",
                        onmouseout="this.style.filter='brightness(0) saturate(100%) invert(91%) sepia(9%) saturate(2661%) hue-rotate(335deg) brightness(60%) contrast(80%)'"
                    ),
                    Button(
                        Img(src='/assets/svgs/User/User_Remove.svg', alt='Remove'),
                        style="background: none; border: none; cursor: pointer; padding: 4px; filter: brightness(0) saturate(80%) invert(16%) sepia(99%) saturate(7444%) hue-rotate(359deg) brightness(40%) contrast(60%);",
                        hx_delete=f"/api/profiles/delete/{item.get('id')}",
                        hx_target="#profilesGrid",
                        hx_indicator="#loading-overlay",
                        hx_trigger="confirmed",
                        onclick=f"""(function(btn) {{
                            const dialog = document.createElement("div");
                            dialog.style.cssText = "position: fixed; top: 0; left: 0; right: 0; bottom: 0; background-color: rgba(0, 0, 0, 0.7); display: flex; justify-content: center; align-items: center; z-index: 1000; opacity: 0; visibility: hidden; transition: opacity 0.3s ease, visibility 0.3s ease;";
                            
                            const content = document.createElement("div");
                            content.style.cssText = "background: #111; padding: 24px; border-radius: 16px; width: 90%; max-width: 500px; position: fixed; left: 50%; top: 50%; transform: translate(-50%, -50%) scale(0.8); transition: all 0.3s cubic-bezier(0.68, -0.55, 0.27, 1.55); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); opacity: 0; visibility: hidden;";
                            
                            content.innerHTML = `
                                <h2 style="margin: 0 0 20px 0; color: #f6cd70;">Delete Profile</h2>
                                <p style="margin: 0 0 24px 0; color: white;">Are you sure you want to delete "{item.get('name', 'this profile')}"?</p>
                                <div style="display: flex; justify-content: flex-end; gap: 12px;">
                                    <button class="cancel" style="padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; background: #333; color: white; transition: all 0.3s ease;">Cancel</button>
                                    <button class="confirm" style="padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; font-weight: 800; background: #f6cd70; color: black; transition: all 0.3s ease;">Delete</button>
                                </div>
                            `;
                            
                            dialog.appendChild(content);
                            document.body.appendChild(dialog);
                            
                            // Trigger reflow to ensure transitions work
                            dialog.offsetHeight;
                            dialog.style.opacity = "1";
                            dialog.style.visibility = "visible";
                            content.style.opacity = "1";
                            content.style.visibility = "visible";
                            content.style.transform = "translate(-50%, -50%) scale(1)";
                            
                            const closeDialog = () => {{
                                dialog.style.opacity = "0";
                                dialog.style.visibility = "hidden";
                                content.style.opacity = "0";
                                content.style.visibility = "hidden";
                                content.style.transform = "translate(-50%, -50%) scale(0.8)";
                                setTimeout(() => document.body.removeChild(dialog), 300);
                            }};
                            
                            dialog.querySelector(".cancel").onclick = closeDialog;
                            dialog.querySelector(".confirm").onclick = () => {{
                                htmx.trigger(btn, "confirmed");
                                closeDialog();
                            }};
                            
                            dialog.onclick = (e) => {{
                                if (e.target === dialog) closeDialog();
                            }};
                            
                            // Add hover effect to buttons
                            const buttons = dialog.querySelectorAll('button');
                            buttons.forEach(button => {{
                                button.onmouseover = () => button.style.transform = 'translateY(-1px)';
                                button.onmouseout = () => button.style.transform = 'none';
                            }});
                        }})(this);""",
                        # Trigger a custom event after successful deletion
                        hx_on__htmx_after_request="if(event.detail.successful) { document.body.dispatchEvent(new CustomEvent('profileDeleted')); }",
                        onmouseover="this.style.filter='brightness(0) saturate(80%) invert(16%) sepia(99%) saturate(7444%) hue-rotate(359deg) brightness(60%) contrast(80%)'",
                        onmouseout="this.style.filter='brightness(0) saturate(80%) invert(16%) sepia(99%) saturate(7444%) hue-rotate(359deg) brightness(40%) contrast(60%)'"
                    ),
                    style=f"padding: 4px 0 4px 16px; {'border-bottom: 1px solid #2a2a2a;' if idx < len(data or []) - 1 else ''} background-color: #000; color: #fff;"
                )
            ) for idx, item in enumerate(data or [])]
        ),
        cls="custom-table",
        style="""width: 100%; border-collapse: separate; border-spacing: 0; 
        border-radius: 8px;
        overflow: hidden; background: #1a1a1a;"""
    )