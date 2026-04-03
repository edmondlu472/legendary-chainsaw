"""Abstract market data provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime

import pandas as pd

from chainsaw.models import OptionContract, OptionQuote, StockQuote


class MarketDataProvider(ABC):
    """Base class for market data providers (separate from broker data)."""

    @abstractmethod
    async def get_stock_quote(self, symbol: str) -> StockQuote:
        """Get current stock quote."""

    @abstractmethod
    async def get_stock_history(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """Get historical OHLCV data. Returns DataFrame with columns: open, high, low, close, volume."""

    @abstractmethod
    async def get_option_chain(self, symbol: str, expiration: date) -> list[OptionQuote]:
        """Get full options chain for a symbol and expiration."""

    @abstractmethod
    async def get_option_history(
        self,
        contract: OptionContract,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Get historical data for a specific option contract."""

    @abstractmethod
    async def get_expirations(self, symbol: str) -> list[date]:
        """Get available expiration dates."""

    @abstractmethod
    async def get_iv_history(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Get historical implied volatility for an underlying."""
