import os
import duckdb
from typing import Optional, Dict
from models.user import User
from uuid import UUID

class UserService:
    def __init__(self):
        self.users_file = os.path.join('datastorage', 'users.csv')
        self.conn = duckdb.connect(':memory:')
        self._init_database()
    
    def _init_database(self):
        """Initialize DuckDB database and import existing data"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR UNIQUE PRIMARY KEY,
                username VARCHAR UNIQUE,
                email VARCHAR UNIQUE,
                sub_account_id VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        if os.path.exists(self.users_file):
            try:
                self.conn.execute(f"""COPY users FROM '{self.users_file}' (
                    DELIMITER ',',
                    HEADER TRUE,
                    QUOTE '"',
                    ESCAPE '"',
                    NULL 'NULL',
                    IGNORE_ERRORS FALSE
                )"""
                )
            except Exception:
                os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
                if not os.path.exists(self.users_file):
                    self.conn.execute(f"COPY users TO '{self.users_file}' (HEADER TRUE)")
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        result = self.conn.execute("""
            SELECT user_id, username, email, sub_account_id
            FROM users
            WHERE user_id = ?
        """, [user_id]).fetchone()
        
        if result:
            return User(
                user_id=UUID(result[0]),
                username=result[1],
                email=result[2],
                sub_account_id=UUID(result[3]) if result[3] else None
            )
        return None
    
    def update_user(self, user: User) -> bool:
        """Update user information"""
        try:
            self.conn.execute("""
                UPDATE users
                SET username = ?,
                    email = ?,
                    sub_account_id = ?
                WHERE user_id = ?
            """, [user.username, user.email, str(user.sub_account_id) if user.sub_account_id else None, str(user.user_id)])
            
            # Save to CSV for persistence
            self.conn.execute(f"COPY users TO '{self.users_file}' (HEADER TRUE)")
            return True
        except Exception as e:
            print(f"Error updating user: {e}")
            return False
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get user statistics"""
        return {
            'total_trades': self.conn.execute("""
                SELECT COUNT(*)
                FROM trades
                WHERE user_id = ?
            """, [user_id]).fetchone()[0],
            'profitable_trades': self.conn.execute("""
                SELECT COUNT(*)
                FROM trades
                WHERE user_id = ? AND profit_loss > 0
            """, [user_id]).fetchone()[0]
        }