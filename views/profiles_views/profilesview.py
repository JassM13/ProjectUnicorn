from fasthtml.common import *

def profiles_view():
    # Sample profile data - In a real app, this would come from a database
    profiles = [
        {"name": "Trading Bot Alpha", "trades": 156, "last_updated": "2h ago", "broker_connected": True},
        {"name": "Swing Trader", "trades": 89, "last_updated": "1d ago", "broker_connected": True},
        {"name": "Day Trading", "trades": 432, "last_updated": "5m ago", "broker_connected": False},
        {"name": "Long Term Portfolio", "trades": 45, "last_updated": "5d ago", "broker_connected": True},
    ]
    
    return Div(
        Script("""
            document.addEventListener('DOMContentLoaded', function() {
                const table = document.querySelector('.profiles-table');
                const headers = table.querySelectorAll('th');
                
                headers.forEach((header, index) => {
                    header.addEventListener('click', () => {
                        const rows = Array.from(table.querySelectorAll('tr:not(:first-child)'));
                        const isNumeric = index === 1; // trades column
                        
                        rows.sort((a, b) => {
                            const aValue = a.children[index].textContent;
                            const bValue = b.children[index].textContent;
                            
                            if (isNumeric) {
                                return parseInt(bValue) - parseInt(aValue);
                            }
                            return bValue.localeCompare(aValue);
                        });
                        
                        rows.forEach(row => table.appendChild(row));
                    });
                });
            });
        """),
        Div(
            H2("Profiles", style="margin: 0 0 24px 0;"),
            Table(
                Tr(
                    Th("Name", style="background: #222; font-weight: 600; text-align: left; padding: 16px; border-bottom: 2px solid #333; cursor: pointer; transition: background-color 0.2s;"),
                    Th("Trades", style="background: #222; font-weight: 600; text-align: left; padding: 16px; border-bottom: 2px solid #333; cursor: pointer; transition: background-color 0.2s;"),
                    Th("Last Updated", style="background: #222; font-weight: 600; text-align: left; padding: 16px; border-bottom: 2px solid #333; cursor: pointer; transition: background-color 0.2s;"),
                    Th("Status", style="background: #222; font-weight: 600; text-align: left; padding: 16px; border-bottom: 2px solid #333; cursor: pointer; transition: background-color 0.2s;"),
                    Th("Actions", style="background: #222; font-weight: 600; text-align: left; padding: 16px; border-bottom: 2px solid #333;"),
                ),
                *[
                    Tr(
                        Td(profile["name"], style="padding: 16px; background-color: #111; border-bottom: 1px solid #333; color: #ccc;"),
                        Td(str(profile["trades"]), style="padding: 16px; background-color: #111; border-bottom: 1px solid #333; color: #ccc;"),
                        Td(profile["last_updated"], style="padding: 16px; background-color: #111; border-bottom: 1px solid #333; color: #ccc;"),
                        Td(
                            Span(
                                "Connected" if profile["broker_connected"] else "Disconnected",
                                style=f"padding: 6px 12px; border-radius: 20px; font-size: 0.9em; font-weight: 500; display: inline-block; {'background: rgba(46, 213, 115, 0.15); color: #2ed573; border: 1px solid rgba(46, 213, 115, 0.3);' if profile['broker_connected'] else 'background: rgba(255, 71, 87, 0.15); color: #ff4757; border: 1px solid rgba(255, 71, 87, 0.3);'}"
                            ),
                            style="padding: 16px; background-color: #111; border-bottom: 1px solid #333; color: #ccc;"
                        ),
                        Td(
                            Div(
                                Button(
                                    Img(src="/assets/svgs/Edit/Edit_Pencil.svg", alt="Edit", style="width: 20px; height: 20px;"),
                                    style="background: none; border: none; cursor: pointer; padding: 8px; transition: all 0.2s ease;"
                                ),
                                Button(
                                    Img(src="/assets/svgs/User/User_Remove.svg", alt="Remove", style="width: 20px; height: 20px;"),
                                    style="background: none; border: none; cursor: pointer; padding: 8px; transition: all 0.2s ease;"
                                ),
                                style="display: flex; align-items: center;"
                            ),
                            style="padding: 16px; background-color: #111; border-bottom: 1px solid #333; color: #ccc;"
                        )
                    )
                    for profile in profiles
                ],
                style="width: 100%; border-collapse: collapse; background: #1a1a1a; border-radius: 8px; overflow: hidden;"
            ),
            style="width: 100%;"
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
            overflow: auto;
        """
    )