import datetime
from fasthtml.common import *
from database.firebase_manager import FirebaseManager
from google.cloud.firestore import FieldFilter
from views.auth_views.register import register_view
from models.user import User
from utils.jwt import generate_token


def register_register_routes(rt):
    firebase_manager = FirebaseManager.getInstance()

    @rt("/register")
    def get(session):
        if 'auth_token' in session:
            return Redirect('/dashboard')
        return register_view()

    @rt("/auth/register")
    def post(session, user: User):
        errors = User.validate(user)
        if errors:
            return Div(errors, id="result", style="color: red;")
        
        try:
            # Check if username or email already exists
            username_query = firebase_manager.db.collection('users').where(filter=FieldFilter('username', '==', user.username)).limit(1).get()
            email_query = firebase_manager.db.collection('users').where(filter=FieldFilter('email', '==', user.email)).limit(1).get()
            
            if len(email_query) > 0:
                return Div(
                    P("Email already exists. ",
                    A("Try logging in instead.", href="/login", style="color: #f6cd70; text-decoration: none;"),
                    style="margin-top: 20px; color: white;"
                    )
                )
            
            if len(username_query) > 0:
                return Div(
                    P("Username already exists. Please choose a different username.",
                    style="margin-top: 20px; color: red;"
                    )
                )
            
            
            
            hashed_password = user.hash_password(plain_password=user.password)
            if not hashed_password:
                return Div(
                    "Password hashing failed",
                    id="result",
                    style="color: red;"
                )
            # Store user data in Firebase
            firebase_manager.db.collection('users').document(user.user_id).set({
                'username': user.username,
                'email': user.email,
                'password_hash': hashed_password,
                'created_at': datetime.datetime.now().timestamp(),
            })
            
            # Set session and generate token
            session['user_id'] = user.user_id
            token = generate_token(user.user_id)
            session['auth_token'] = token
            
            return Redirect('/dashboard')
        except Exception as e:
            print(f"Firebase error: {str(e)}")
            return Div(
                "An error occurred during registration",
                id="result",
                style="color: red;"
            )

    return rt