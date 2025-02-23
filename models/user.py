from dataclasses import dataclass
from typing import Optional
import uuid

@dataclass
class User:
    user_id: str = str(uuid.uuid4())
    identifier: str = ""  # Can be either username or email
    username: str = ""
    email: str = ""
    password: str = ""
    sub_account_id: Optional[uuid.UUID] = None
    
    def __post_init__(self):
        # Set identifier based on username or email if not provided
        if not self.identifier:
            self.identifier = self.email if self.email else self.username
    
    @classmethod
    def validate(cls, user) -> tuple[str]:
        """Validate user data and return a tuple of (is_valid, error_message)"""
        errors = []
        
        if not user.identifier:
            errors.append("Please provide username or email")
            
        if '@' in user.identifier:
            user.email = user.identifier
        else:
            user.username = user.identifier
            
        if user.username and len(user.username) < 3:
            errors.append("Username must be at least 3 characters long")
        
        if user.email and '@' not in user.email:
            errors.append("Invalid email address")
            
        if not user.password or len(user.password) < 8:
            errors.append("Password must be at least 8 characters long")
            
        return '; '.join(errors) if errors else None