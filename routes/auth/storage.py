import duckdb
import hashlib
import os
import uuid
from typing import Optional
from models.user import User

class UserStorage:
    def __init__(self, file_path: str = 'datastorage/users.csv'):
        self.file_path = file_path
        self.conn = duckdb.connect(':memory:')
        self._init_database()
    
    def _init_database(self):
        # Create users table if it doesn't exist
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username VARCHAR PRIMARY KEY,
                email VARCHAR UNIQUE,
                password_hash VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Import existing data from CSV if it exists
        if os.path.exists(self.file_path):
            self.conn.execute(f"COPY users FROM '{self.file_path}' (DELIMITER ',', HEADER TRUE, QUOTE '""', NULL_PADDING TRUE)")
    
    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, user: User) -> bool:
        # Check if user already exists
        existing_user = self.conn.execute("""
            SELECT 1 FROM users
            WHERE username = ? OR email = ?
        """, [user.username, user.email]).fetchone()
        
        if existing_user:
            return False
            
        try:
            password_hash = self._hash_password(user.password)
            user.user_id = str(uuid.uuid4())
            self.conn.execute("""
                INSERT INTO users (user_id, username, email, password_hash)
                VALUES (?, ?, ?, ?)
            """, [user.user_id, user.username, user.email, password_hash])
            # Save to CSV for persistence
            self.conn.execute(f"COPY users TO '{self.file_path}' (HEADER TRUE)")
            return True
        except Exception:
            return False
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        result = self.conn.execute("""
            SELECT user_id, username, email, password_hash
            FROM users
            WHERE username = ?
        """, [username]).fetchone()
        
        if result:
            return User(identifier=result[1], username=result[1], email=result[2], password="", user_id=result[0])
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        result = self.conn.execute("""
            SELECT user_id, username, email, password_hash
            FROM users
            WHERE email = ?
        """, [email]).fetchone()
        
        if result:
            return User(identifier=result[1], username=result[1], email=result[2], password="", user_id=result[0])
        return None
    
    def verify_password(self, identifier: str, password: str, is_email: bool = False) -> bool:
        user = self.get_user_by_email(identifier) if is_email else self.get_user_by_username(identifier)
        if not user:
            return False
        
        # Get stored password hash
        result = self.conn.execute("""
            SELECT password_hash
            FROM users
            WHERE username = ? OR email = ?
        """, [user.username, user.email]).fetchone()
        
        if not result:
            return False
            
        stored_hash = result[0]
        return stored_hash == self._hash_password(password)