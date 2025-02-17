from dataclasses import dataclass

@dataclass
class User:
    username: str
    email: str
    password: str

def validate_user(user: User):
    errors = []
    if len(user.username) < 3:
        errors.append("Username must be at least 3 characters long")
    if '@' not in user.email:
        errors.append("Invalid email address")
    if len(user.password) < 8:
        errors.append("Password must be at least 8 characters long")
    return errors