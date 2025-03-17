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

def calendar_view(session):
    cal, month_name, year, today = get_calendar_data()
    month = today.month
    
    # Calendar navigation and header
    header = Div(
        Div(
            H2(f"{month_name} {year}", style="margin: 0;"),
            style="flex: 1; text-align: left;"
        ),
        Div(
            Button("Previous", 
                   onclick=f"?month={today.month-1 if today.month>1 else 12}&year={year if today.month>1 else year-1}",
                   style="background: #2a2a2a; border: none; color: white; padding: 8px 16px; border-radius: 20px; margin-right: 10px; cursor: pointer; transition: all 0.3s ease; width: 100px;",
                   onmouseover="this.style.backgroundColor='#3a3a3a'",
                   onmouseout="this.style.backgroundColor='#2a2a2a'"),
            Button("Next", 
                   onclick=f"?month={today.month+1 if today.month<12 else 1}&year={year if today.month<12 else year+1}",
                   style="background: #2a2a2a; border: none; color: white; padding: 8px 16px; border-radius: 20px; cursor: pointer; transition: all 0.3s ease; width: 100px;",
                   onmouseover="this.style.backgroundColor='#3a3a3a'",
                   onmouseout="this.style.backgroundColor='#2a2a2a'"),
            style="display: flex; justify-content: flex-end; width: 220px;"
        ),
        style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;"
    )
    
    # Weekday header
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_header = Div(
        *[Div(day, style="width: 40px; text-align: center; font-weight: bold;") for day in weekdays],
        style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 10px; margin-bottom: 10px;"
    )
    
    # Get previous and next month dates
    prev_month = month - 1 if month > 1 else 12
    prev_year = year if month > 1 else year - 1
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    
    # Get the last few days of the previous month
    _, last_day_prev = calendar.monthrange(prev_year, prev_month)
    first_weekday, _ = calendar.monthrange(year, month)
    prev_month_days = [last_day_prev - first_weekday + i + 1 for i in range(first_weekday)]
    
    # Get the first few days of the next month
    next_month_days = list(range(1, 15))  # More than enough days for filling the calendar
    
    # Calendar grid
    calendar_cells = []
    prev_month_idx = 0
    next_month_idx = 0
    
    for week in cal:
        for day in week:
            if day == 0:
                # Add previous/next month dates
                if prev_month_idx < len(prev_month_days):
                    day_num = prev_month_days[prev_month_idx]
                    prev_month_idx += 1
                else:
                    day_num = next_month_days[next_month_idx]
                    next_month_idx += 1
                
                day_style = f"""
                    width: 100%; height: 100%; display: flex;
                    align-items: flex-start; justify-content: flex-start;
                    border-radius: 12px; cursor: pointer;
                    transition: all 0.3s ease; position: relative;
                    padding: 15px; opacity: 0.3;
                """
                calendar_cells.append(Div(str(day_num), style=day_style))
            else:
                is_today = (day == today.day and today.month == datetime.now().month and today.year == datetime.now().year)
                day_style = f"""
                    width: 100%; height: 100%; display: flex;
                    align-items: flex-start; justify-content: flex-start;
                    border-radius: 12px; cursor: pointer;
                    {'background-color: #2a2a2a; box-shadow: 0 4px 12px rgba(255,255,255,0.1);' if is_today else ''}
                    transition: all 0.3s ease; position: relative;
                    padding: 15px;
                """
                calendar_cells.append(
                    Div(str(day),
                        onclick=f"updateSelectedDate('{month_name}', {day}, {year}, this)",
                        onmouseover="this.style.backgroundColor='#2a2a2a'; this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(255,255,255,0.1)'",
                        onmouseout=f"this.style.backgroundColor='{'#2a2a2a' if is_today else 'transparent'}'; this.style.transform='translateY(0)'; this.style.boxShadow='{'0 4px 12px rgba(255,255,255,0.1)' if is_today else 'none'}'",
                        style=day_style
                    )
                )
    
    calendar_grid = Div(
        *calendar_cells,
        style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 15px; margin-top: 10px; flex-grow: 1;"
    )
    
    # Left side: calendar
    left_side = Div(
        header,
        weekday_header,
        calendar_grid,
        style="flex: 2; display: flex; flex-direction: column;"
    )
    
    # Right side: daily profit / news
    right_side = Div(
        H2("Selected Date", id="selected-date-header", style="margin-bottom: 10px;"),
        Div("No date selected", id="selected-date-content", style="margin-bottom: 20px;"),
        H2("Day's Profit", style="margin-bottom: 10px;"),
        Div("Profit details go here...", style="margin-bottom: 20px;"),
        H2("Events", style="margin-bottom: 10px;"),
        Div("No events scheduled", style="margin-bottom: 20px;"),
        style="flex: 1; padding: 20px;"
    )
    
    # JavaScript for handling date selection
    script = Script("""
        function updateSelectedDate(month, day, year, element) {
            document.getElementById('selected-date-header').textContent = `${month} ${day}, ${year}`;
            document.getElementById('selected-date-content').textContent = `You selected ${month} ${day}, ${year}`;
            
            // Remove previous selection highlight
            document.querySelectorAll('.selected-date').forEach(el => {
                el.classList.remove('selected-date');
                el.style.backgroundColor = 'transparent';
            });
            
            // Add highlight to selected date while maintaining alignment
            element.classList.add('selected-date');
            element.style.backgroundColor = '#3a3a3a';
        }
    """)
    
    # Main container with black background
    container = Div(
        left_side,
        right_side,
        script,
        style="""
            display: flex; padding: 30px; color: white; 
            height: 95vh; border-radius: 16px; position: absolute; 
            right: 20px; top: 20px; left: 100px; 
            bottom: 20px; border: 1px solid rgba(255, 255, 255, 0.1);
        """
    )
    
    return container
