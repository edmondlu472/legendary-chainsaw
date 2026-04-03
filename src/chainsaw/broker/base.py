"""Abstract broker interface. All broker integrations implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from chainsaw.models import (
    OptionContract,
    OptionQuote,
    Order,
    PortfolioSnapshot,
    Position,
    StockQuote,
)


class Broker(ABC):
    """Base class for all broker integrations."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the broker."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close broker connection."""

    @abstractmethod
    async def get_stock_quote(self, symbol: str) -> StockQuote:
        """Get current quote for a stock."""

    @abstractmethod
    async def get_option_chain(
        self,
        symbol: str,
        expiration: date | None = None,
    ) -> list[OptionQuote]:
        """Get options chain for a symbol, optionally filtered by expiration."""

    @abstractmethod
    async def get_option_quote(self, contract: OptionContract) -> OptionQuote:
        """Get current quote for a specific option contract."""

    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """Submit an order and return it with updated status/ID."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if cancellation was accepted."""

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Get current status of an order."""

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get all current positions."""

    @abstractmethod
    async def get_portfolio(self) -> PortfolioSnapshot:
        """Get full portfolio snapshot including positions, margin, and cash."""

    @abstractmethod
    async def get_expirations(self, symbol: str) -> list[date]:
        """Get available option expiration dates for a symbol."""
