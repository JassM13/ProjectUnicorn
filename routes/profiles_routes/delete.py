from fasthtml.common import *
from database.firebase_manager import FirebaseManager
from middleware.authorized_request import authorized_request

def register_delete_profile_routes(rt):
    firebase_manager = FirebaseManager.getInstance()

    @rt("/api/profiles/{profile_id}")
    @authorized_request
    async def delete_profile(session, request, profile_id: str):
        try:
            # Get user_id from session
            user_id = session.get('user_id')
            if not user_id:
                return Div(
                    "You must be logged in to delete a profile",
                    style="background-color: #f44336; color: white; padding: 10px; border-radius: 5px;"
                )
            
            # Get the profile document
            profile_ref = firebase_manager.db.collection('profiles').document(profile_id)
            profile = profile_ref.get()
            
            # Check if profile exists and belongs to the user
            if not profile.exists:
                return Div(
                    "Profile not found",
                    style="background-color: #f44336; color: white; padding: 10px; border-radius: 5px;"
                )
            
            if profile.to_dict()['user_id'] != user_id:
                return Div(
                    "You don't have permission to delete this profile",
                    style="background-color: #f44336; color: white; padding: 10px; border-radius: 5px;"
                )
            
            # Delete the profile
            profile_ref.delete()
            
            return Div(
                "Profile deleted successfully!",
                style="background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;"
            )
        except Exception as e:
            return Div(
                f"Error deleting profile: {str(e)}",
                style="background-color: #f44336; color: white; padding: 10px; border-radius: 5px;"
            )

    return rt