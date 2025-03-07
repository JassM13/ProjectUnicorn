from fasthtml.common import *
from database.firebase_manager import FirebaseManager
from google.cloud.firestore import FieldFilter
from views.main_view.mainview import mainview
from utils.jwt import generate_token
from views.profiles_views.profilesview import profiles_view
from models.user import User
from models.sub_account import SubAccount
from middleware.authorized_request import authorized_request
import bcrypt
import json
import uuid
from datetime import datetime

def register_profile_routes(rt):
    firebase_manager = FirebaseManager.getInstance()

    @rt("/profiles")
    @authorized_request
    def get(session):
        return mainview(active="profiles")
    
    @rt("/api/profiles")
    @authorized_request
    async def create_profile(session, request):
        # Validate the session and save to database
        try:
            # Get user_id from session
            user_id = session.get('user_id')
            if not user_id:
                return Div(
                    "You must be logged in to create a profile",
                    style="background-color: #f44336; color: white; padding: 10px; border-radius: 5px;"
                )
            
            # Extract form data
            form_data = await request.form()
            profile_name = form_data.get('profile_name')
            broker_account = form_data.get('broker_account', False)
            
            if not profile_name:
                return Div(
                    "Profile name is required",
                    style="background-color: #f44336; color: white; padding: 10px; border-radius: 5px;"
                )
                
            # Create a new profile
            profile_id = str(uuid.uuid4())
            
            # Save to the database
            firebase_manager.db.collection('profiles').document(profile_id).set({
                'profile_name': profile_name,
                'broker_account': broker_account == 'true' or broker_account == True,
                'created_at': datetime.now(),
                'user_id': user_id,
                'trades': 0,
                'last_updated': datetime.now()
            })
            
            # Return success message with styling
            return Div(
                "Profile created successfully!",
                style="background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;"
            )
        except Exception as e:
            # Return error message with styling
            return Div(
                f"Error creating profile: {str(e)}",
                style="background-color: #f44336; color: white; padding: 10px; border-radius: 5px;"
            )
    
    @rt("/api/profiles/get")
    @authorized_request
    def get_profiles(session):
        # Fetch profiles from database if user is logged in
        profiles = []
        if session and session.get('user_id'):
            user_id = session.get('user_id')
            
            # Query profiles collection for this user
            profile_docs = firebase_manager.db.collection('profiles').where('user_id', '==', user_id).get()
            
            for doc in profile_docs:
                profile_data = doc.to_dict()
                
                # Format the last_updated timestamp
                last_updated_timestamp = profile_data.get('last_updated')
                last_updated = 'never'
                
                if last_updated_timestamp:
                    # Calculate time difference
                    now = datetime.now()
                    if isinstance(last_updated_timestamp, str):
                        last_updated = last_updated_timestamp
                    else:
                        # Only calculate diff if timestamp is a datetime object
                        try:
                            # Check if it's a Firebase timestamp object
                            if hasattr(last_updated_timestamp, 'timestamp'):
                                # Convert Firebase timestamp to datetime
                                last_updated_timestamp = last_updated_timestamp.timestamp()
                                last_updated_timestamp = datetime.fromtimestamp(last_updated_timestamp)
                            
                            # Now calculate the time difference
                            diff = now - last_updated_timestamp
                            
                            if diff.days > 0:
                                last_updated = f"{diff.days}d ago"
                            elif diff.seconds // 3600 > 0:
                                last_updated = f"{diff.seconds // 3600}h ago"
                            elif diff.seconds // 60 > 0:
                                last_updated = f"{diff.seconds // 60}m ago"
                            else:
                                last_updated = "just now"
                        except Exception as e:
                            # Log the error for debugging
                            print(f"Error formatting timestamp: {str(e)}, type: {type(last_updated_timestamp)}")
                            # If there's any error in calculation, use a default value
                            last_updated = "unknown"
                
                profiles.append({
                    "id": doc.id,
                    "name": profile_data.get('profile_name', 'Unnamed Profile'),
                    "trades": profile_data.get('trades', 0),
                    "last_updated": last_updated,
                    "broker_account": profile_data.get('broker_account', False)
                })
        
        # Return profiles as JSON
        return json.dumps(profiles)

    return rt