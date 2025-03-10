from fasthtml.common import *
from database.firebase_manager import FirebaseManager
from middleware.authorized_request import authorized_request
import uuid
from datetime import datetime

def register_create_profile_routes(rt):
    firebase_manager = FirebaseManager.getInstance()

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

    return rt