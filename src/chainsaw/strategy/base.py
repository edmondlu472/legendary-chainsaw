"""Abstract strategy interface and signal types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from chainsaw.models import Order, PortfolioSnapshot


class SignalType(Enum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    ADJUST = "ADJUST"
    HOLD = "HOLD"


@dataclass
class Signal:
    signal_type: SignalType
    orders: list[Order] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)


class Strategy(ABC):
    """Base class for all trading strategies.

    Strategies receive portfolio state and market data, then produce signals
    (lists of orders to execute). The execution engine handles submission
    and fill management.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.enabled = True

    @abstractmethod
    async def evaluate(self, portfolio: PortfolioSnapshot) -> Signal:
        """Evaluate current market conditions and portfolio state.

        Returns a Signal indicating what action (if any) to take.
        This is called on every tick/interval by the execution loop.
        """

    @abstractmethod
    async def on_fill(self, order: Order) -> None:
        """Called when an order from this strategy is filled.

        Use this to update internal state, adjust stops, etc.
        """

    async def on_start(self) -> None:
        """Called once when the strategy is activated."""

    async def on_stop(self) -> None:
        """Called once when the strategy is deactivated."""
