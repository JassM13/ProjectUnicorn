from typing import Optional
from uuid import UUID
from models.sub_account import SubAccount
from database.postgresql_manager import PostgresManager

class SubAccountService:
    def __init__(self):
        self.pg_manager = PostgresManager.getInstance()
        self._init_database()
    
    def _init_database(self):
        """Initialize PostgreSQL database"""
        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sub_accounts (
                        id VARCHAR PRIMARY KEY,
                        user_id VARCHAR NOT NULL,
                        name VARCHAR NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                """)
                conn.commit()
        finally:
            self.pg_manager.release_connection(conn)
    
    def create_sub_account(self, user_id: str, name: str) -> Optional[SubAccount]:
        """Create a new sub-account for a user"""
        conn = self.pg_manager.get_connection()
        try:
            # First verify if user exists
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT user_id FROM users WHERE user_id = %s
                """, [user_id])
                if not cur.fetchone():
                    print(f"Error: User with id {user_id} does not exist")
                    return None
                
                sub_account = SubAccount(name=name)
                cur.execute("""
                    INSERT INTO sub_accounts (id, user_id, name)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, [str(sub_account.id), user_id, name])
                conn.commit()
                return sub_account
        except Exception as e:
            print(f"Error creating sub-account: {str(e)}")
            conn.rollback()
            return None
        finally:
            self.pg_manager.release_connection(conn)
    
    def get_user_sub_accounts(self, user_id: str) -> list[SubAccount]:
        """Get all sub-accounts for a user"""
        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, name
                    FROM sub_accounts
                    WHERE user_id = %s
                """, [user_id])
                
                sub_accounts = []
                for row in cur.fetchall():
                    sub_accounts.append(SubAccount(
                        id=UUID(row[0]),
                        name=row[1]
                    ))
                return sub_accounts
        finally:
            self.pg_manager.release_connection(conn)
    
    def get_default_sub_account(self, user_id: str) -> Optional[SubAccount]:
        """Get or create default sub-account for a user"""
        sub_accounts = self.get_user_sub_accounts(user_id)
        if not sub_accounts:
            print(f"No sub-accounts found for user {user_id}")
            return self.create_sub_account(user_id, "Default Account")
        return sub_accounts[0]  # Return the first sub-account as default