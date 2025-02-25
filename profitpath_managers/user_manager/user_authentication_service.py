import hashlib
import os
import uuid
from typing import Optional
from models.user import User
from database.postgresql_manager import PostgresManager

class UserAuthenticationService:
    def __init__(self, file_path: str = 'datastorage/users.csv'):
        self.file_path = file_path
        self.pg_manager = PostgresManager.getInstance()
        self._init_database()
    
    def _init_database(self):
        """Initialize PostgreSQL database and import existing data"""
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
                
                if os.path.exists(self.file_path):
                    try:
                        with open(self.file_path, 'r') as f:
                            cur.copy_expert(
                                "COPY users FROM STDIN WITH (FORMAT CSV, HEADER TRUE)",
                                f
                            )
                    except Exception:
                        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
                        if not os.path.exists(self.file_path):
                            with open(self.file_path, 'w') as f:
                                cur.copy_expert(
                                    "COPY users TO STDIN WITH (FORMAT CSV, HEADER TRUE)",
                                    f
                                )
                conn.commit()
        finally:
            self.pg_manager.release_connection(conn)
    
    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, user: User) -> bool:
        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                # Check if user already exists
                cur.execute("""
                    SELECT 1 FROM users
                    WHERE username = %s OR email = %s
                """, [user.username, user.email])
                existing_user = cur.fetchone()
                
                if existing_user:
                    return False
                    
                password_hash = self._hash_password(user.password)
                user.user_id = str(uuid.uuid4())
                cur.execute("""
                    INSERT INTO users (user_id, username, email, password_hash)
                    VALUES (%s, %s, %s, %s)
                """, [user.user_id, user.username, user.email, password_hash])
                
                # Save to CSV for persistence
                with open(self.file_path, 'w') as f:
                    cur.copy_expert(
                        "COPY users TO STDIN WITH (FORMAT CSV, HEADER TRUE)",
                        f
                    )
                conn.commit()
                return True
        except Exception:
            conn.rollback()
            return False
        finally:
            self.pg_manager.release_connection(conn)
    
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