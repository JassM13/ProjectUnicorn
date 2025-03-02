import hashlib
import uuid
from typing import Optional
from models.user import User
from database.postgresql_manager import PostgresManager
from .sub_account_service import SubAccountService

class UserAuthenticationService:
    def __init__(self):
        self.pg_manager = PostgresManager.getInstance()
        self.sub_account_service = SubAccountService()
        self._init_database()
    
    def _init_database(self):
        """Initialize PostgreSQL database"""
        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id VARCHAR PRIMARY KEY,
                        username VARCHAR UNIQUE,
                        email VARCHAR UNIQUE,
                        password_hash VARCHAR,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        finally:
            self.pg_manager.release_connection(conn)
    
    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT user_id, username, email, password_hash
                    FROM users
                    WHERE username = %s
                """, [username])
                result = cur.fetchone()
                
                if result:
                    return User(identifier=result[1], username=result[1], email=result[2], password="", user_id=result[0])
                return None
        finally:
            self.pg_manager.release_connection(conn)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT user_id, username, email, password_hash
                    FROM users
                    WHERE email = %s
                """, [email])
                result = cur.fetchone()
                
                if result:
                    return User(identifier=result[1], username=result[1], email=result[2], password="", user_id=result[0])
                return None
        finally:
            self.pg_manager.release_connection(conn)
    
    def verify_password(self, identifier: str, password: str, is_email: bool = False) -> bool:
        user = self.get_user_by_email(identifier) if is_email else self.get_user_by_username(identifier)
        if not user:
            return False
        
        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                # Get stored password hash
                cur.execute("""
                    SELECT password_hash
                    FROM users
                    WHERE username = %s OR email = %s
                """, [user.username, user.email])
                result = cur.fetchone()
                
                if not result:
                    return False
                    
                stored_hash = result[0]
                return stored_hash == self._hash_password(password)
        finally:
            self.pg_manager.release_connection(conn)
    def register_user(self, user: User) -> Optional[User]:
        """Register a new user with username, email and password"""
            
        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                # Check if user already exists
                cur.execute("""
                    SELECT 1 FROM users
                    WHERE username = %s OR email = %s
                """, [user.username, user.email])
                if cur.fetchone():
                    print(f"User already exists with username {user.username} or email {user.email}")
                    return None
                
                # Hash the password using SHA-256
                password_hash = self._hash_password(user.password)
                
                # Insert the new user
                cur.execute("""
                    INSERT INTO users (user_id, username, email, password_hash)
                    VALUES (%s, %s, %s, %s)
                """, [user.user_id, user.username, user.email, password_hash])
                conn.commit()  # Commit the user creation first
                
                # Create default sub-account for the new user
                default_account = self.sub_account_service.create_sub_account(user.user_id, "Default Account")
                if default_account:
                    user.sub_account_id = default_account.id
                    print(f"Default sub-account created for user: {user.username}")
                else:
                    print(f"Failed to create default sub-account for user: {user.username}")
                    conn.rollback()
                    return None
                
                conn.commit()
                print(f"User registered successfully: {user.username}")
                return user
        except Exception as e:
            print(f"Error registering user: {str(e)}")
            conn.rollback()
            return None
        finally:
            self.pg_manager.release_connection(conn)