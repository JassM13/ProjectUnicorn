import os
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
            self.trades_file = os.path.join('datastorage', 'trades.csv')
            self.pg_manager = PostgresManager.getInstance()
            self._init_database()
            self._initialized = True

    def _init_database(self):
        """Initialize PostgreSQL database and import existing data"""
        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id VARCHAR,
                        user_id VARCHAR,
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
                
                if os.path.exists(self.trades_file):
                    try:
                        with open(self.trades_file, 'r') as f:
                            cur.copy_expert(
                                "COPY trades FROM STDIN WITH (FORMAT CSV, HEADER TRUE)",
                                f
                            )
                    except Exception:
                        os.makedirs(os.path.dirname(self.trades_file), exist_ok=True)
                        if not os.path.exists(self.trades_file):
                            with open(self.trades_file, 'w') as f:
                                cur.copy_expert(
                                    "COPY trades TO STDIN WITH (FORMAT CSV, HEADER TRUE)",
                                    f
                                )
                conn.commit()
        finally:
            self.pg_manager.release_connection(conn)

    def add_trade(self, user_id: str, trade_data: Dict) -> bool:
        """Add a new trade to the database"""
        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trades (id, user_id, symbol, entry_price, exit_price, 
                                    position_size, entry_date, exit_date, profit_loss, 
                                    trade_type, risk_reward_ratio)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, [
                    trade_data.get('id'),
                    user_id,
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
                
                # Save to CSV for persistence
                with open(self.trades_file, 'w') as f:
                    cur.copy_expert(
                        "COPY trades TO STDIN WITH (FORMAT CSV, HEADER TRUE)",
                        f
                    )
                conn.commit()
                return True
        except Exception as e:
            conn.rollback()
            print(f"Error adding trade: {e}")
            return False
        finally:
            self.pg_manager.release_connection(conn)

    def get_user_trades(self, user_id: str) -> List[Dict]:
        """Get all trades for a specific user"""
        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, symbol, entry_price, exit_price, position_size,
                           entry_date, exit_date, profit_loss, trade_type, risk_reward_ratio
                    FROM trades
                    WHERE user_id = %s
                    ORDER BY entry_date DESC
                """, [user_id])
                
                trades = []
                for row in cur.fetchall():
                    trades.append({
                        'id': row[0],
                        'symbol': row[1],
                        'entry_price': row[2],
                        'exit_price': row[3],
                        'position_size': row[4],
                        'entry_date': row[5],
                        'exit_date': row[6],
                        'profit_loss': row[7],
                        'trade_type': row[8],
                        'risk_reward_ratio': row[9]
                    })
                return trades
        finally:
            self.pg_manager.release_connection(conn)