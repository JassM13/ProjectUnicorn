from typing import Dict
from database.postgresql_manager import PostgresManager

class TradeStatistics:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.pg_manager = PostgresManager.getInstance()
            self._initialized = True

    def calculate_stats(self, user_id: str) -> Dict:
        """Calculate trading statistics for a user"""
        conn = self.pg_manager.get_connection()
        try:
            with conn.cursor() as cur:
                # Get total trades
                cur.execute("""
                    SELECT COUNT(*), 
                           COUNT(CASE WHEN profit_loss > 0 THEN 1 END),
                           AVG(position_size),
                           AVG(risk_reward_ratio)
                    FROM trades
                    WHERE user_id = %s
                """, [user_id])
                
                total_trades, winning_trades, avg_position_size, avg_risk_reward = cur.fetchone()
                
                return {
                    'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                    'avg_position_size': avg_position_size or 0,
                    'risk_reward': avg_risk_reward or 0,
                    'total_trades': total_trades
                }
        finally:
            self.pg_manager.release_connection(conn)