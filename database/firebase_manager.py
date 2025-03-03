import os
import base64
import json
from firebase_admin import credentials, initialize_app, firestore
from dotenv import load_dotenv

class FirebaseManager:
    __instance = None

    @staticmethod
    def getInstance():
        if FirebaseManager.__instance is None:
            FirebaseManager.__instance = FirebaseManager()
        return FirebaseManager.__instance

    def __init__(self):
        if FirebaseManager.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            try:
                # Read base64 credentials from environment variable
                cred_base64 = os.getenv('FIREBASE_CREDENTIALS_BASE64')
                if not cred_base64:
                    raise Exception("FIREBASE_CREDENTIALS_BASE64 environment variable not found")
                
                # Decode and parse the credentials
                cred_json = base64.b64decode(cred_base64).decode('utf-8')
                cred_dict = json.loads(cred_json)
                cred = credentials.Certificate(cred_dict)
                initialize_app(cred)
                self.db = firestore.client()
            except Exception as e:
                raise Exception(f"Failed to initialize Firebase: {str(e)}")
    
    def close(self):
        """Cleanup method (if needed when shutting down)."""
        pass  # Firebase handles cleanup automatically
