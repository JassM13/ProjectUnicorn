from typing import Optional, Dict
from uuid import UUID
from models.user import User
from database.postgresql_manager import PostgresManager

class UserProfileService:
    def __init__(self):
        self.pg_manager = PostgresManager.getInstance()
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
                        sub_account_id VARCHAR,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        finally:
            self.pg_manager.release_connection(conn)
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT user_id, username, email, sub_account_id
                    FROM users
                    WHERE user_id = %s
                """, [user_id])
                result = cur.fetchone()
                
                if result:
                    return User(
                        user_id=UUID(result[0]),
                        username=result[1],
                        email=result[2],
                        sub_account_id=UUID(result[3]) if result[3] else None
                    )
                return None
        finally:
            self.pg_manager.release_connection(conn)
    
    def update_user(self, user: User) -> bool:
        """Update user information"""
        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE users
                    SET username = %s,
                        email = %s,
                        sub_account_id = %s
                    WHERE user_id = %s
                """, [user.username, user.email, str(user.sub_account_id) if user.sub_account_id else None, str(user.user_id)])
                conn.commit()
                return True
        except Exception as e:
            conn.rollback()
            print(f"Error updating user: {e}")
            return False
        finally:
            self.pg_manager.release_connection(conn)
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get user statistics"""
        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*)
                    FROM trades
                    WHERE user_id = %s
                """, [user_id])
                total_trades = cur.fetchone()[0]
                
                cur.execute("""
                    SELECT COUNT(*)
                    FROM trades
                    WHERE user_id = %s AND profit_loss > 0
                """, [user_id])
                profitable_trades = cur.fetchone()[0]
                
                return {
                    'total_trades': total_trades,
                    'profitable_trades': profitable_trades
                }
        finally:
            self.pg_manager.release_connection(conn)