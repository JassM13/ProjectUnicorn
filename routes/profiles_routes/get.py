from fasthtml.common import *
from database.firebase_manager import FirebaseManager
from google.cloud.firestore import FieldFilter
from middleware.authorized_request import authorized_request
from datetime import datetime
import json
from views.profiles_views.gridding.grid_table import create_grid_table

def register_get_profile_routes(rt):
    firebase_manager = FirebaseManager.getInstance()

    @rt("/api/profiles/get")
    @authorized_request
    def get_profiles(session, request=None):
        profiles = []
        
        if not session or not session.get('user_id'):
            error_json = json.dumps({"error": "Authentication required"})
            return Response(error_json, content_type="application/json")
            
        user_id = session.get('user_id')
        
        # Query profiles collection for this user
        profile_docs = firebase_manager.db.collection('profiles').where(filter=FieldFilter('user_id', '==', user_id)).get()
        
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
        
        # Check for API request vs HTMX request
        is_htmx_request = request and request.headers.get('HX-Request') == 'true'
        content_type = request and request.headers.get('Accept')
        
        if is_htmx_request or (content_type and 'text/html' in content_type):
            # Return HTML grid for HTMX requests
            return create_grid_table(profiles)
        else:
            # Return JSON for API requests
            json_response = json.dumps(profiles)
            # Create a Response object with the appropriate content-type
            return Response(json_response, content_type="application/json")
        
    return rt