from dataclasses import dataclass, field
from typing import List, Optional
from uuid import UUID, uuid4
#from .linked_broker_account import LinkedBrokerAccount
from .trade import TradeGroup

@dataclass
class SubAccount:
    name: str
    id: UUID = field(default_factory=uuid4)
    #linked_broker_account: Optional[LinkedBrokerAccount] = None
    trade_groups: List[TradeGroup] = field(default_factory=list)