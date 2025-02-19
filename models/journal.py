from dataclasses import dataclass, field
from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime
from .sub_account import SubAccount
from .journal_entry import JournalEntry

@dataclass
class Journal:
    id: UUID = field(default_factory=uuid4)
    title: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    account: Optional[SubAccount] = None
    entries: List[JournalEntry] = field(default_factory=list)

