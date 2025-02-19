from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime
from .journal import Journal
from .trade import TradeGroup

@dataclass
class JournalEntry:
    id: UUID = field(default_factory=uuid4)
    content: Optional[bytes] = None
    created_at: datetime = field(default_factory=datetime.now)
    journal: Optional[Journal] = None
    trade_groups: Optional[TradeGroup] = None