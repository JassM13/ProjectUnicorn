import os
import duckdb
from datetime import datetime
from typing import List, Dict, Optional
from uuid import UUID

class TradeService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._trades = []
        return cls._instance

    def __init__(self):
        # Initialize only if it hasn't been initialized
        if not hasattr(self, '_trades'):
            self._trades = []

    def add_trade(self, trade):
        self._trades.append(trade)
        return True

    def get_trades(self, user_id=None):
        return self._trades

    def calculate_stats(self, user_id=None):
        return {
            'win_rate': 68.5,
            'avg_position_size': 5420,
            'risk_reward': 2.5,
            'total_trades': len(self._trades)
        }
    def _init_database(self):
        """Initialize DuckDB database and import existing data"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id VARCHAR,
                user_id VARCHAR,
                symbol VARCHAR,
                entry_price DOUBLE,
                exit_price DOUBLE,
                position_size DOUBLE,
                entry_date TIMESTAMP,
                exit_date TIMESTAMP,
                profit_loss DOUBLE,
                trade_type VARCHAR,
                risk_reward_ratio DOUBLE
            )
        """)
        
        if os.path.exists(self.trades_file):
            try:
                self.conn.execute(f"""COPY trades FROM '{self.trades_file}' (
                    DELIMITER ',',
                    HEADER TRUE,
                    QUOTE '"',
                    ESCAPE '"',
                    NULL 'NULL',
                    IGNORE_ERRORS FALSE
                )"""
                )
            except Exception:
                # If file doesn't exist or is empty, create it
                os.makedirs(os.path.dirname(self.trades_file), exist_ok=True)
                if not os.path.exists(self.trades_file):
                    self.conn.execute(f"COPY trades TO '{self.trades_file}' (HEADER TRUE)")
    
    def add_trade(self, user_id: str, trade_data: Dict) -> bool:
        """Add a new trade to the database"""
        try:
            self.conn.execute("""
                INSERT INTO trades (id, user_id, symbol, entry_price, exit_price, 
                                  position_size, entry_date, exit_date, profit_loss, 
                                  trade_type, risk_reward_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            self.conn.execute(f"COPY trades TO '{self.trades_file}' (HEADER TRUE)")
            return True
        except Exception as e:
            print(f"Error adding trade: {e}")
            return False
    
    def get_user_trades(self, user_id: str) -> List[Dict]:
        """Get all trades for a specific user"""
        try:
            result = self.conn.execute("""
                SELECT * FROM trades
                WHERE user_id = ?
                ORDER BY entry_date DESC
            """, [user_id]).fetchall()
            
            columns = ['id', 'user_id', 'symbol', 'entry_price', 'exit_price',
                      'position_size', 'entry_date', 'exit_date', 'profit_loss',
                      'trade_type', 'risk_reward_ratio']
            
            return [dict(zip(columns, row)) for row in result]
        except Exception as e:
            print(f"Error getting trades: {e}")
            return []
    
    def calculate_stats(self, user_id: str) -> Dict:
        """Calculate trading statistics for a user"""
        trades = self.get_user_trades(user_id)
        if not trades:
            return {
                'win_rate': 0,
                'avg_position_size': 0,
                'risk_reward': 0,
                'total_trades': 0
            }
        
        winning_trades = sum(1 for trade in trades if float(trade['profit_loss']) > 0)
        total_trades = len(trades)
        position_sizes = [float(trade['position_size']) for trade in trades]
        risk_rewards = [float(trade['risk_reward_ratio']) for trade in trades if trade['risk_reward_ratio']]
        
        return {
            'win_rate': round((winning_trades / total_trades) * 100, 1) if total_trades > 0 else 0,
            'avg_position_size': round(sum(position_sizes) / len(position_sizes), 2) if position_sizes else 0,
            'risk_reward': round(sum(risk_rewards) / len(risk_rewards), 1) if risk_rewards else 0,
            'total_trades': total_trades
        }