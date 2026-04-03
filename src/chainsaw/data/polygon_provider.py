"""Polygon.io market data provider implementation."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
from polygon import RESTClient

from chainsaw.config import PolygonConfig
from chainsaw.data.provider import MarketDataProvider
from chainsaw.logging import get_logger
from chainsaw.models import (
    Greeks,
    OptionContract,
    OptionQuote,
    OptionType,
    StockQuote,
)

log = get_logger(__name__)


class PolygonDataProvider(MarketDataProvider):
    """Polygon.io data provider."""

    def __init__(self, config: PolygonConfig | None = None) -> None:
        self._config = config or PolygonConfig()
        self._client = RESTClient(api_key=self._config.api_key)

    async def get_stock_quote(self, symbol: str) -> StockQuote:
        snapshot = self._client.get_snapshot_ticker("stocks", symbol)
        return StockQuote(
            symbol=symbol,
            bid=snapshot.ticker.last_quote.bid_price if snapshot.ticker.last_quote else 0.0,
            ask=snapshot.ticker.last_quote.ask_price if snapshot.ticker.last_quote else 0.0,
            last=snapshot.ticker.last_trade.price if snapshot.ticker.last_trade else 0.0,
            volume=int(snapshot.ticker.day.volume) if snapshot.ticker.day else 0,
            timestamp=datetime.now(),
        )

    async def get_stock_history(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        multiplier, span = _parse_timeframe(timeframe)
        aggs = list(self._client.list_aggs(
            ticker=symbol,
            multiplier=multiplier,
            timespan=span,
            from_=start.isoformat(),
            to=end.isoformat(),
            limit=50000,
        ))

        rows = []
        for bar in aggs:
            rows.append({
                "timestamp": datetime.fromtimestamp(bar.timestamp / 1000),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
        return df

    async def get_option_chain(self, symbol: str, expiration: date) -> list[OptionQuote]:
        contracts = list(self._client.list_snapshot_options_chain(
            symbol,
            params={
                "expiration_date": expiration.isoformat(),
            },
        ))

        quotes: list[OptionQuote] = []
        for snap in contracts:
            details = snap.details
            opt_type = OptionType.CALL if details.contract_type == "call" else OptionType.PUT

            contract = OptionContract(
                symbol=symbol,
                expiration=date.fromisoformat(details.expiration_date),
                strike=details.strike_price,
                option_type=opt_type,
            )

            greeks = Greeks()
            if snap.greeks:
                greeks = Greeks(
                    delta=snap.greeks.delta or 0.0,
                    gamma=snap.greeks.gamma or 0.0,
                    theta=snap.greeks.theta or 0.0,
                    vega=snap.greeks.vega or 0.0,
                )
            if snap.implied_volatility:
                greeks.iv = snap.implied_volatility

            quotes.append(OptionQuote(
                contract=contract,
                bid=snap.last_quote.bid if snap.last_quote else 0.0,
                ask=snap.last_quote.ask if snap.last_quote else 0.0,
                last=snap.last_trade.price if snap.last_trade else 0.0,
                volume=int(snap.day.volume) if snap.day else 0,
                open_interest=int(snap.open_interest) if snap.open_interest else 0,
                greeks=greeks,
            ))

        return quotes

    async def get_option_history(
        self,
        contract: OptionContract,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        ticker = _build_option_ticker(contract)
        aggs = list(self._client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start.isoformat(),
            to=end.isoformat(),
            limit=50000,
        ))

        rows = []
        for bar in aggs:
            rows.append({
                "timestamp": datetime.fromtimestamp(bar.timestamp / 1000),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
        return df

    async def get_expirations(self, symbol: str) -> list[date]:
        contracts = list(self._client.list_options_contracts(
            underlying_ticker=symbol,
            limit=1000,
        ))
        expirations = sorted({date.fromisoformat(c.expiration_date) for c in contracts})
        return expirations

    async def get_iv_history(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        # Polygon doesn't have a direct IV history endpoint.
        # Use VIX as proxy for SPX, or compute from option chain snapshots.
        log.warning("iv_history_not_native", msg="Using VIX proxy for IV history")
        return await self.get_stock_history("VIX" if symbol == "SPY" else symbol, start, end)


def _parse_timeframe(timeframe: str) -> tuple[int, str]:
    mapping = {
        "1m": (1, "minute"),
        "5m": (5, "minute"),
        "15m": (15, "minute"),
        "1h": (1, "hour"),
        "1d": (1, "day"),
        "1w": (1, "week"),
    }
    return mapping.get(timeframe, (1, "day"))


def _build_option_ticker(contract: OptionContract) -> str:
    """Build Polygon option ticker format: O:SPY251219C00450000"""
    t = "C" if contract.option_type == OptionType.CALL else "P"
    exp = contract.expiration.strftime("%y%m%d")
    strike = f"{int(contract.strike * 1000):08d}"
    return f"O:{contract.symbol}{exp}{t}{strike}"
