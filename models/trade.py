from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4
from sub_account import SubAccount
from journal_entry import JournalEntry

@dataclass
class Trade:
    contract_name: str
    entered_at: datetime
    exited_at: datetime
    instrument_type: str
    trade_day: datetime
    type: str
    id: UUID = field(default_factory=uuid4)
    entry_price: float = 0.0
    exit_price: float = 0.0
    fees: float = 0.0
    pnl: float = 0.0
    size: float = 0.0
    trade_group: Optional['TradeGroup'] = None

@dataclass
class TradeGroup:
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    is_manually_grouped: bool = False
    trades: List[Trade] = field(default_factory=list)
    account: Optional['SubAccount'] = None
    journal_entry: Optional['JournalEntry'] = None

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