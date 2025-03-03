from fasthtml.common import *
from database.firebase_manager import FirebaseManager
from google.cloud.firestore import FieldFilter
from utils.jwt import generate_token
from views.auth_views.login import login_view
from models.user import User
import bcrypt

def register_login_routes(rt):
    firebase_manager = FirebaseManager.getInstance()

    @rt("/login")
    def get(session):
        if 'auth_token' in session:
            return Redirect('/dashboard')
        return login_view()

    @rt("/auth/login")
    def post(session, user: User):
        errors = User.validate(user)
        if errors:
            return Div(errors, id="result", style="color: red;")
        
        print(user.identifier)
        # Try to verify password with either username or email
        try:
            if '@' in user.identifier:
                # Query user by email
                user_query = firebase_manager.db.collection('users').where(filter=FieldFilter('email', '==', user.identifier)).limit(1).get()
            else:
                # Query user by username
                user_query = firebase_manager.db.collection('users').where(filter=FieldFilter('username', '==', user.identifier)).limit(1).get()

            if not user_query:
                return Div(
                    P("User does not exist. ",
                    A("Try signing up instead.", href="/register", style="color: #f6cd70; text-decoration: none;"),
                    style="margin-top: 20px; color: white;"
                    )
                )

            user_data = user_query[0].to_dict()
            stored_password = user_data.get('password_hash')

            if not bcrypt.checkpw(user.password.encode('utf-8'), stored_password):
                return Div(
                    P("Invalid Credentials",
                    style="margin-top: 20px; color: red;"
                    )
                )

            # Generate token using the document ID as user_id
            token = generate_token(user_query[0].id)
            session['auth_token'] = token
            session['user_id'] = user_query[0].id
            
            return Redirect('/dashboard')
            
        except Exception as e:
            print(f"Login error: {str(e)}")
            return Div("An error occurred during login", id="result", style="color: red;")

    return rt