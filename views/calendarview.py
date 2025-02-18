from fasthtml.common import *
from datetime import datetime, timedelta
import calendar

def get_calendar_data(year=None, month=None):
    if year is None or month is None:
        today = datetime.now()
        year = today.year
        month = today.month
    else:
        today = datetime.now()
    
    cal = calendar.monthcalendar(year, month)
    month_name = calendar.month_name[month]
    return cal, month_name, year, today

def calendar_view():
    cal, month_name, year, today = get_calendar_data()
    
    # Calendar navigation and header
    header = Div(
        Button("<", 
               onclick=f"?month={today.month-1 if today.month>1 else 12}&year={year if today.month>1 else year-1}",
               style="background: none; border: none; color: white; font-size: 20px; cursor: pointer;"),
        H2(f"{month_name} {year}", style="margin: 0 20px;"),
        Button(">", 
               onclick=f"?month={today.month+1 if today.month<12 else 1}&year={year if today.month<12 else year+1}",
               style="background: none; border: none; color: white; font-size: 20px; cursor: pointer;"),
        style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;"
    )
    
    # Weekday header
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_header = Div(
        *[Div(day, style="width: 40px; text-align: center; font-weight: bold;") for day in weekdays],
        style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 10px; margin-bottom: 10px;"
    )
    
    # Calendar grid
    calendar_cells = []
    for week in cal:
        for day in week:
            if day == 0:
                calendar_cells.append(Div(style="width: 40px; height: 40px;"))
            else:
                is_today = (day == today.day and today.month == datetime.now().month and today.year == datetime.now().year)
                day_style = f"""
                    width: 40px; height: 50px; display: flex; align-items: center;
                    justify-content: center; border-radius: 12px; cursor: pointer;
                    {'background-color: #2a2a2a; box-shadow: 0 4px 12px rgba(255,255,255,0.1);' if is_today else ''}
                    transition: all 0.3s ease; position: relative;
                """
                calendar_cells.append(
                    Div(str(day),
                        onclick=f"alert('Selected: {month_name} {day}, {year}')",
                        onmouseover="this.style.backgroundColor='#2a2a2a'; this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(255,255,255,0.1)'",
                        onmouseout=f"this.style.backgroundColor='{'#2a2a2a' if is_today else 'transparent'}'; this.style.transform='translateY(0)'; this.style.boxShadow='{'0 4px 12px rgba(255,255,255,0.1)' if is_today else 'none'}'",
                        style=day_style
                    )
                )
    
    calendar_grid = Div(
        *calendar_cells,
        style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 15px; margin-top: 10px;"
    )
    
    # Left side: calendar
    left_side = Div(
        header,
        weekday_header,
        calendar_grid,
        style="flex: 2;"
    )
    
    # Right side: daily profit / news
    right_side = Div(
        H2("Day's Profit", style="margin-bottom: 10px;"),
        Div("Profit details go here...", style="margin-bottom: 20px;"),
        H2("News & Events", style="margin-bottom: 10px;"),
        Div("Upcoming news/events listed here...", style="margin-bottom: 20px;"),
        style="flex: 1; padding: 20px;"
    )
    
    # Main container with black background
    container = Div(
        left_side,
        right_side,
        style="""
            display: flex; 
            padding: 30px; 
            background-color: #000; 
            color: white; 
            height: 95vh; 
            border-radius: 24px; 
            position: absolute; 
            right: 20px; 
            top: 20px; 
            left: 100px; 
            bottom: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        """
    )
    
    return container
