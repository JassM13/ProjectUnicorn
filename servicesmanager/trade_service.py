import csv
import os
from datetime import datetime
from typing import List, Dict, Optional

class TradeService:
    def __init__(self):
        self.trades_file = os.path.join('datastorage', 'trades.csv')
        self._ensure_trades_file_exists()
    
    def _ensure_trades_file_exists(self):
        """Ensure trades.csv exists with headers"""
        if not os.path.exists(self.trades_file):
            with open(self.trades_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'user_id', 'symbol', 'entry_price', 'exit_price', 
                                'position_size', 'entry_date', 'exit_date', 'profit_loss', 
                                'trade_type', 'risk_reward_ratio'])
    
    def add_trade(self, user_id: str, trade_data: Dict) -> bool:
        """Add a new trade to the CSV file"""
        try:
            with open(self.trades_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
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
            return True
        except Exception as e:
            print(f"Error adding trade: {e}")
            return False
    
    def get_user_trades(self, user_id: str) -> List[Dict]:
        """Get all trades for a specific user"""
        trades = []
        try:
            with open(self.trades_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['user_id'] == user_id:
                        trades.append(row)
            return trades
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