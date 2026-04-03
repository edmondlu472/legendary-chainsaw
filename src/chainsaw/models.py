"""Core domain models for the trading platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OptionType(Enum):
    CALL = "CALL"
    PUT = "PUT"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class OptionContract:
    symbol: str
    expiration: date
    strike: float
    option_type: OptionType
    multiplier: int = 100

    @property
    def dte(self) -> int:
        return (self.expiration - date.today()).days

    def __str__(self) -> str:
        t = "C" if self.option_type == OptionType.CALL else "P"
        return f"{self.symbol} {self.expiration} {self.strike}{t}"


@dataclass
class Greeks:
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    iv: float = 0.0


@dataclass
class OptionQuote:
    contract: OptionContract
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    greeks: Greeks = field(default_factory=Greeks)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class StockQuote:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass
class Position:
    contract: OptionContract | str  # OptionContract for options, str symbol for stock
    quantity: int
    avg_cost: float
    market_price: float = 0.0
    greeks: Greeks = field(default_factory=Greeks)

    @property
    def market_value(self) -> float:
        multiplier = self.contract.multiplier if isinstance(self.contract, OptionContract) else 1
        return self.quantity * self.market_price * multiplier

    @property
    def unrealized_pnl(self) -> float:
        multiplier = self.contract.multiplier if isinstance(self.contract, OptionContract) else 1
        return self.quantity * (self.market_price - self.avg_cost) * multiplier


@dataclass
class Order:
    contract: OptionContract | str
    side: Side
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    filled_quantity: int = 0
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None


@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    net_liquidation: float
    cash: float
    margin_used: float
    margin_available: float
    positions: list[Position] = field(default_factory=list)

    @property
    def margin_utilization(self) -> float:
        total = self.margin_used + self.margin_available
        return self.margin_used / total if total > 0 else 0.0

    @property
    def total_delta(self) -> float:
        return sum(p.greeks.delta * p.quantity for p in self.positions)

    @property
    def total_gamma(self) -> float:
        return sum(p.greeks.gamma * p.quantity for p in self.positions)

    @property
    def total_theta(self) -> float:
        return sum(p.greeks.theta * p.quantity for p in self.positions)

    @property
    def total_vega(self) -> float:
        return sum(p.greeks.vega * p.quantity for p in self.positions)
