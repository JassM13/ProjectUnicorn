from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import uuid

@dataclass
class Trade:
    symbol: str
    entered_at: Optional[datetime]
    exited_at: Optional[datetime]
    instrument_type: str
    trade_day: Optional[datetime]
    type: str
    id: uuid.UUID = uuid.uuid4()
    entry_price: float = 0.0
    exit_price: float = 0.0
    fees: Optional[float] = 0.0
    pnl: float = 0.0
    size: float = 0.0
    trade_group: Optional['TradeGroup'] = None
    
    def __post_init__(self):
        # Convert string dates to datetime objects
        for date_field in ['entered_at', 'exited_at', 'trade_day']:
            value = getattr(self, date_field)
            if isinstance(value, str):
                try:
                    setattr(self, date_field, datetime.fromisoformat(value.replace('Z', '+00:00')))
                except ValueError:
                    setattr(self, date_field, datetime.now())
            elif value is None:
                setattr(self, date_field, datetime.now())
        
        # Convert numeric fields to float
        for float_field in ['entry_price', 'exit_price', 'fees', 'pnl', 'size']:
            value = getattr(self, float_field)
            if value is not None:
                setattr(self, float_field, float(value or 0.0))
        
        # Calculate PNL if not provided
        if not self.pnl:
            self.calculate_pnl()
    
    def calculate_pnl(self):
        """Calculate PNL based on position type, size, and prices"""
        if self.type.lower() == 'long':
            self.pnl = (self.exit_price - self.entry_price) * self.size - self.fees
        else:  # short
            self.pnl = (self.entry_price - self.exit_price) * self.size - self.fees

@dataclass
class TradeGroup:
    id: uuid.UUID = uuid.uuid4()
    created_at: datetime = field(default_factory=datetime.now)
    is_manually_grouped: bool = False
    trades: List[Trade] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'is_manually_grouped': self.is_manually_grouped,
            'account_id': str(self.account.id) if self.account else None,
            'journal_entry_id': str(self.journal_entry.id) if self.journal_entry else None,
            'trades': [str(trade.id) for trade in self.trades]
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict())