from fasthtml.common import *
from database.firebase_manager import FirebaseManager
from google.cloud.firestore import FieldFilter
from views.main_view.mainview import mainview
from utils.jwt import generate_token
from views.profiles_views.profilesview import profiles_view
from models.user import User
import bcrypt

def register_profile_routes(rt):
    firebase_manager = FirebaseManager.getInstance()

    @rt("/profiles")
    def get(session):
        return mainview(active="profiles")

    return rt