from fasthtml.common import *
from profitpath_managers.user_manager.user_authentication_service import UserAuthenticationService
from views.auth_views.register import register_view
from models.user import User

def register_register_routes(rt):
    auth_service = UserAuthenticationService()

    @rt("/register")
    def get(session):
        if 'AuthToken' in session:
            return Redirect('/dashboard')
        return register_view()

    @rt("/auth/register")
    def post(session, user: User):
        errors = User.validate(user)
        print(f"errors: {errors}")
        if errors:
            return Div(errors, id="result", style="color: red;")
        
        if not auth_service.register_user(user):
            return Div(
                "Username or email already exists",
                id="result",
                style="color: red;"
            )
        session['user_id'] = user.user_id

        Redirect('/dashboard')

    return rt