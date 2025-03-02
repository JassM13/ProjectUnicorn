from typing import List, Dict, Optional
from uuid import UUID
from database.postgresql_manager import PostgresManager

class TradeOperations:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.pg_manager = PostgresManager.getInstance()
            self._init_database()
            self._initialized = True

    def _init_database(self):
        """Initialize PostgreSQL database"""
        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id VARCHAR PRIMARY KEY,
                        sub_account_id VARCHAR REFERENCES sub_accounts(id),
                        trade_group VARCHAR,
                        symbol VARCHAR,
                        entry_price DOUBLE PRECISION,
                        exit_price DOUBLE PRECISION,
                        position_size DOUBLE PRECISION,
                        entry_date TIMESTAMP,
                        exit_date TIMESTAMP,
                        profit_loss DOUBLE PRECISION,
                        trade_type VARCHAR,
                        risk_reward_ratio DOUBLE PRECISION
                    )
                """)
                conn.commit()
        finally:
            self.pg_manager.release_connection(conn)

    def add_trade(self, user_id: str, sub_account_id: str, trade_data: Dict) -> bool:
        """Add a new trade to the database under a specific sub-account"""
        from profitpath_managers.user_manager.sub_account_service import SubAccountService

        if not user_id:
            print("Error: user_id is required")
            return False

        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                # First, try to get the sub-account if provided
                if sub_account_id:
                    cur.execute("""
                        SELECT user_id FROM sub_accounts WHERE id = %s
                    """, [sub_account_id])
                    result = cur.fetchone()
                    
                    # Verify the sub-account belongs to the user
                    if result and result[0] != user_id:
                        print("Error: Sub-account does not belong to the user")
                        return False
                else:
                    result = None

                if not result:
                    # If sub_account doesn't exist or wasn't provided, get/create default sub-account
                    sub_account_service = SubAccountService()
                    default_sub_account = sub_account_service.get_default_sub_account(user_id)
                    if not default_sub_account:
                        print("Error: Unable to create default sub-account")
                        return False
                    sub_account_id = str(default_sub_account.id)

                cur.execute("""
                    INSERT INTO trades (id, sub_account_id, trade_group, symbol, entry_price, 
                                    exit_price, position_size, entry_date, exit_date, 
                                    profit_loss, trade_type, risk_reward_ratio)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, [
                    trade_data.get('id'),
                    sub_account_id,
                    trade_data.get('trade_group'),
                    trade_data.get('symbol'),
                    trade_data.get('entry_price'),
                    trade_data.get('exit_price'),
                    trade_data.get('position_size'),
                    trade_data.get('entry_date'),
                    trade_data.get('exit_date'),
                    trade_data.get('profit_loss'),
                    trade_data.get('trade_type'),
                    trade_data.get('risk_reward_ratio')
                ])
                conn.commit()
                return True
        except Exception as e:
            conn.rollback()
            print(f"Error adding trade: {e}")
            return False
        finally:
            self.pg_manager.release_connection(conn)

    def get_user_trades(self, user_id: str) -> List[Dict]:
        """Get all trades for a specific user through their sub-accounts"""
        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT t.id, t.trade_group, t.symbol, t.entry_price, t.exit_price, 
                           t.position_size, t.entry_date, t.exit_date, t.profit_loss, 
                           t.trade_type, t.risk_reward_ratio, s.name as sub_account_name
                    FROM trades t
                    JOIN sub_accounts s ON t.sub_account_id = s.id
                    WHERE s.user_id = %s
                    ORDER BY t.entry_date DESC
                """, [user_id])
                
                trades = []
                for row in cur.fetchall():
                    trades.append({
                        'id': row[0],
                        'trade_group': row[1],
                        'symbol': row[2],
                        'entry_price': row[3],
                        'exit_price': row[4],
                        'position_size': row[5],
                        'entry_date': row[6],
                        'exit_date': row[7],
                        'profit_loss': row[8],
                        'trade_type': row[9],
                        'risk_reward_ratio': row[10],
                        'sub_account_name': row[11]
                    })
                return trades
        finally:
            self.pg_manager.release_connection(conn)