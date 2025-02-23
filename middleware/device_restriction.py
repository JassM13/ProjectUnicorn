from functools import wraps
from fasthtml.common import *

def restrict_small_devices(min_width=768):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            return Div(
                Script(f"""
                    if (window.innerWidth < {min_width}) {{
                        window.location.href = '/device-restricted';
                    }}
                    window.addEventListener('resize', function() {{
                        if (window.innerWidth < {min_width}) {{
                            window.location.href = '/device-restricted';
                        }}
                    }});
                """),
                view_func(*args, **kwargs)
            )
        return wrapper
    return decorator