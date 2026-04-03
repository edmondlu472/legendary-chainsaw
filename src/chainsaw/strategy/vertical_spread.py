"""Vertical spread strategy — bull call spreads and bear put spreads."""

from __future__ import annotations

from datetime import date

from chainsaw.broker.base import Broker
from chainsaw.data.provider import MarketDataProvider
from chainsaw.logging import get_logger
from chainsaw.models import (
    OptionContract,
    OptionQuote,
    OptionType,
    Order,
    OrderType,
    PortfolioSnapshot,
    Side,
)
from chainsaw.pricing.engine import PricingEngine
from chainsaw.strategy.base import Signal, SignalType, Strategy

log = get_logger(__name__)


class VerticalSpreadStrategy(Strategy):
    """Opens vertical spreads based on directional signals and IV conditions.

    Bull call spread: Buy lower strike call, sell higher strike call.
    Bear put spread: Buy higher strike put, sell lower strike put.
    """

    def __init__(
        self,
        symbols: list[str],
        broker: Broker,
        data: MarketDataProvider,
        pricing: PricingEngine,
        max_dte: int = 45,
        min_dte: int = 14,
        spread_width: float = 5.0,
        min_iv_percentile: float = 0.3,
        max_risk_per_trade: float = 500.0,
    ) -> None:
        super().__init__(name="vertical_spread")
        self.symbols = symbols
        self.broker = broker
        self.data = data
        self.pricing = pricing
        self.max_dte = max_dte
        self.min_dte = min_dte
        self.spread_width = spread_width
        self.min_iv_percentile = min_iv_percentile
        self.max_risk_per_trade = max_risk_per_trade

    async def evaluate(self, portfolio: PortfolioSnapshot) -> Signal:
        for symbol in self.symbols:
            signal = await self._evaluate_symbol(symbol, portfolio)
            if signal.signal_type != SignalType.HOLD:
                return signal

        return Signal(signal_type=SignalType.HOLD, reason="No setups found")

    async def _evaluate_symbol(self, symbol: str, portfolio: PortfolioSnapshot) -> Signal:
        try:
            quote = await self.data.get_stock_quote(symbol)
            expirations = await self.data.get_expirations(symbol)
        except Exception as e:
            log.warning("data_fetch_failed", symbol=symbol, error=str(e))
            return Signal(signal_type=SignalType.HOLD, reason=f"Data unavailable for {symbol}")

        # Find suitable expiration
        target_exp = self._select_expiration(expirations)
        if not target_exp:
            return Signal(signal_type=SignalType.HOLD, reason=f"No suitable expiration for {symbol}")

        # Get option chain
        chain = await self.data.get_option_chain(symbol, target_exp)
        if not chain:
            return Signal(signal_type=SignalType.HOLD, reason=f"Empty chain for {symbol}")

        # Simple directional signal: bullish if price above short-term midpoint
        # This is a placeholder — replace with real signal logic
        direction = self._get_direction_signal(quote.last)
        if direction is None:
            return Signal(signal_type=SignalType.HOLD, reason="No directional signal")

        # Build the spread
        orders = self._build_spread(symbol, quote.last, target_exp, chain, direction)
        if not orders:
            return Signal(signal_type=SignalType.HOLD, reason="Could not construct spread")

        return Signal(
            signal_type=SignalType.OPEN,
            orders=orders,
            reason=f"{'Bull call' if direction == 'bull' else 'Bear put'} spread on {symbol}",
            confidence=0.6,
        )

    def _select_expiration(self, expirations: list[date]) -> date | None:
        today = date.today()
        for exp in sorted(expirations):
            dte = (exp - today).days
            if self.min_dte <= dte <= self.max_dte:
                return exp
        return None

    def _get_direction_signal(self, price: float) -> str | None:
        """Placeholder directional signal. Replace with real analysis."""
        # TODO: Implement actual signal logic (momentum, mean reversion, etc.)
        return "bull"

    def _build_spread(
        self,
        symbol: str,
        spot: float,
        expiration: date,
        chain: list[OptionQuote],
        direction: str,
    ) -> list[Order]:
        if direction == "bull":
            return self._build_bull_call_spread(symbol, spot, expiration, chain)
        else:
            return self._build_bear_put_spread(symbol, spot, expiration, chain)

    def _build_bull_call_spread(
        self,
        symbol: str,
        spot: float,
        expiration: date,
        chain: list[OptionQuote],
    ) -> list[Order]:
        calls = [q for q in chain if q.contract.option_type == OptionType.CALL and q.bid > 0]
        calls.sort(key=lambda q: q.contract.strike)

        # Find ATM call (closest to spot)
        long_quote = min(calls, key=lambda q: abs(q.contract.strike - spot), default=None)
        if not long_quote:
            return []

        # Short leg is spread_width above
        target_short_strike = long_quote.contract.strike + self.spread_width
        short_quote = min(
            calls,
            key=lambda q: abs(q.contract.strike - target_short_strike),
            default=None,
        )
        if not short_quote or short_quote.contract.strike <= long_quote.contract.strike:
            return []

        # Calculate spread cost and check risk
        spread_cost = long_quote.ask - short_quote.bid
        max_contracts = int(self.max_risk_per_trade / (spread_cost * 100)) if spread_cost > 0 else 0
        quantity = max(1, min(max_contracts, 5))

        log.info(
            "bull_call_spread",
            symbol=symbol,
            long_strike=long_quote.contract.strike,
            short_strike=short_quote.contract.strike,
            cost=spread_cost,
            qty=quantity,
        )

        return [
            Order(
                contract=long_quote.contract,
                side=Side.BUY,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                limit_price=long_quote.ask,
            ),
            Order(
                contract=short_quote.contract,
                side=Side.SELL,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                limit_price=short_quote.bid,
            ),
        ]

    def _build_bear_put_spread(
        self,
        symbol: str,
        spot: float,
        expiration: date,
        chain: list[OptionQuote],
    ) -> list[Order]:
        puts = [q for q in chain if q.contract.option_type == OptionType.PUT and q.bid > 0]
        puts.sort(key=lambda q: q.contract.strike)

        long_quote = min(puts, key=lambda q: abs(q.contract.strike - spot), default=None)
        if not long_quote:
            return []

        target_short_strike = long_quote.contract.strike - self.spread_width
        short_quote = min(
            puts,
            key=lambda q: abs(q.contract.strike - target_short_strike),
            default=None,
        )
        if not short_quote or short_quote.contract.strike >= long_quote.contract.strike:
            return []

        spread_cost = long_quote.ask - short_quote.bid
        max_contracts = int(self.max_risk_per_trade / (spread_cost * 100)) if spread_cost > 0 else 0
        quantity = max(1, min(max_contracts, 5))

        return [
            Order(
                contract=long_quote.contract,
                side=Side.BUY,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                limit_price=long_quote.ask,
            ),
            Order(
                contract=short_quote.contract,
                side=Side.SELL,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                limit_price=short_quote.bid,
            ),
        ]

    async def on_fill(self, order: Order) -> None:
        log.info("spread_fill", contract=str(order.contract), side=order.side.value, price=order.fill_price)
