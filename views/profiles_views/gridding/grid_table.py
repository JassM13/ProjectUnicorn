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
                Td(item.get('name', ''), style="padding: 4px 0 4px 16px; border-bottom: 1px solid #2a2a2a; background-color: #000; color: #fff;"),
                Td(str(item.get('trades', 0)), style="padding: 4px 0 4px 16px; border-bottom: 1px solid #2a2a2a; background-color: #000; color: #fff;"),
                Td(item.get('last_updated', ''), style="padding: 4px 0 4px 16px; border-bottom: 1px solid #2a2a2a; background-color: #000; color: #fff;"),
                Td('True' if item.get('broker_account') else 'False', style="padding: 4px 0 4px 16px; border-bottom: 1px solid #2a2a2a; background-color: #000; color: #fff;"),
                Td(
                    Button(
                        Img(src='/assets/svgs/Edit/Edit_Pencil.svg', alt='Edit'),
                        style="background: none; border: none; cursor: pointer; padding: 4px;",
                        hx_get=f"/api/profiles/{item.get('id')}/edit",
                        hx_target="#profile_form"
                    ),
                    Button(
                        Img(src='/assets/svgs/User/User_Remove.svg', alt='Remove'),
                        style="background: none; border: none; cursor: pointer; padding: 4px;",
                        hx_delete=f"/api/profiles/{item.get('id')}",
                        hx_target="#profilesGrid",
                        hx_confirm="Are you sure you want to delete this profile?"
                    ),
                    style="padding: 4px 0 4px 16px; border-bottom: 1px solid #2a2a2a; background-color: #000; color: #fff;"
                )
            ) for item in (data or [])]
        ),
        cls="custom-table",
        style="width: 100%; border-collapse: separate; border-spacing: 0; border-radius: 8px; overflow: hidden; background: #1a1a1a;"
    )