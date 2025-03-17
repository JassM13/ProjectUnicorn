from middleware.authorized_request import authorized_request
from views.main_view.mainview import mainview
from database.firebase_manager import FirebaseManager
from google.cloud.firestore import FieldFilter

def register_account_routes(rt):
    @authorized_request
    @rt("/account")
    def get(session):
        # Get Firebase instance
        firebase_manager = FirebaseManager.getInstance()
        
        # Get user data from Firebase using session user_id
        if session and session.get('user_id'):
            user_doc = firebase_manager.db.collection('users').document(session['user_id']).get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                # Create a session user object with required attributes
                session['user'] = {
                    'username': user_data.get('username'),
                    'email': user_data.get('email'),
                    'user_id': session['user_id']
                }
        
        return mainview(session, active="account")
    
    return rt