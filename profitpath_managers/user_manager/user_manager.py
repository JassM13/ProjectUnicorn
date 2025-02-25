from typing import Optional, Dict
from models.user import User
from .user_authentication_service import UserAuthenticationService
from .user_profile_service import UserProfileService

class UserManager:
    def __init__(self):
        self.storage = UserAuthenticationService()
        self.service = UserProfileService()
    
    def create_user(self, user: User) -> bool:
        """Create a new user with both authentication and profile data"""
        # First create the auth record
        if not self.storage.create_user(user):
            return False
            
        # Then create/update the profile
        return self.service.update_user(user)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.storage.get_user_by_username(username)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.storage.get_user_by_email(email)
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.service.get_user_by_id(user_id)
    
    def verify_password(self, identifier: str, password: str, is_email: bool = False) -> bool:
        """Verify user password"""
        return self.storage.verify_password(identifier, password, is_email)
    
    def update_user(self, user: User) -> bool:
        """Update user information"""
        return self.service.update_user(user)
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get user statistics"""
        return self.service.get_user_stats(user_id)