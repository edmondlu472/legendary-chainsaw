"""Iron condor strategy — sell OTM call spread + OTM put spread for premium."""

from __future__ import annotations

from datetime import date, timedelta

from chainsaw.broker.base import Broker
from chainsaw.data.provider import MarketDataProvider
from chainsaw.logging import get_logger
from chainsaw.models import (
    OptionQuote,
    OptionType,
    Order,
    OrderType,
    PortfolioSnapshot,
    Side,
)
from chainsaw.pricing.engine import PricingEngine
from chainsaw.signals.composite import CompositeSignalGenerator
from chainsaw.strategy.base import Signal, SignalType, Strategy

log = get_logger(__name__)


class IronCondorStrategy(Strategy):
    """Sells iron condors on high-IV underlyings for premium income.

    Structure:
    - Buy OTM put (wing)
    - Sell OTM put (closer to ATM)
    - Sell OTM call (closer to ATM)
    - Buy OTM call (wing)

    Profits when underlying stays within the short strikes.
    """

    def __init__(
        self,
        symbols: list[str],
        broker: Broker,
        data: MarketDataProvider,
        pricing: PricingEngine,
        target_dte: int = 30,
        min_dte: int = 20,
        max_dte: int = 50,
        short_delta: float = 0.20,
        wing_width: float = 5.0,
        min_credit: float = 0.50,
        max_risk_per_trade: float = 500.0,
        min_iv_rank: float = 30.0,
        signal_generator: CompositeSignalGenerator | None = None,
        lookback_days: int = 252,
    ) -> None:
        super().__init__(name="iron_condor")
        self.symbols = symbols
        self.broker = broker
        self.data = data
        self.pricing = pricing
        self.target_dte = target_dte
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.short_delta = short_delta
        self.wing_width = wing_width
        self.min_credit = min_credit
        self.max_risk_per_trade = max_risk_per_trade
        self.min_iv_rank = min_iv_rank
        self.signals = signal_generator or CompositeSignalGenerator(min_iv_rank_for_selling=min_iv_rank)
        self.lookback_days = lookback_days

    async def evaluate(self, portfolio: PortfolioSnapshot) -> Signal:
        for symbol in self.symbols:
            signal = await self._evaluate_symbol(symbol, portfolio)
            if signal.signal_type != SignalType.HOLD:
                return signal

        return Signal(signal_type=SignalType.HOLD, reason="No iron condor setups")

    async def _evaluate_symbol(self, symbol: str, portfolio: PortfolioSnapshot) -> Signal:
        try:
            quote = await self.data.get_stock_quote(symbol)
            expirations = await self.data.get_expirations(symbol)
        except Exception as e:
            log.warning("data_error", symbol=symbol, error=str(e))
            return Signal(signal_type=SignalType.HOLD)

        target_exp = self._select_expiration(expirations)
        if not target_exp:
            return Signal(signal_type=SignalType.HOLD, reason="No suitable expiration")

        chain = await self.data.get_option_chain(symbol, target_exp)
        if not chain:
            return Signal(signal_type=SignalType.HOLD, reason="Empty chain")

        # Get IV history for rank/percentile calculation
        avg_iv = self._avg_chain_iv(chain)
        try:
            end = date.today()
            start = end - timedelta(days=self.lookback_days)
            iv_history = await self.data.get_iv_history(symbol, start, end)
            if not iv_history.empty and "close" in iv_history.columns:
                iv_series = iv_history["close"]
                if not self.signals.should_sell_premium(avg_iv, iv_series):
                    return Signal(signal_type=SignalType.HOLD, reason=f"{symbol}: IV rank too low for premium selling")
            else:
                # Fallback: use chain IV directly
                if avg_iv < self.min_iv_rank / 100:
                    return Signal(signal_type=SignalType.HOLD, reason=f"IV too low ({avg_iv:.1%})")
        except Exception as e:
            log.warning("iv_history_unavailable", symbol=symbol, error=str(e))
            if avg_iv < self.min_iv_rank / 100:
                return Signal(signal_type=SignalType.HOLD, reason=f"IV too low ({avg_iv:.1%})")

        orders = self._build_iron_condor(symbol, quote.last, target_exp, chain)
        if not orders:
            return Signal(signal_type=SignalType.HOLD, reason="Could not build condor")

        return Signal(
            signal_type=SignalType.OPEN,
            orders=orders,
            reason=f"Iron condor on {symbol} (IV: {avg_iv:.1%})",
            confidence=0.5,
        )

    def _select_expiration(self, expirations: list[date]) -> date | None:
        today = date.today()
        best = None
        best_diff = float("inf")
        for exp in expirations:
            dte = (exp - today).days
            if self.min_dte <= dte <= self.max_dte:
                diff = abs(dte - self.target_dte)
                if diff < best_diff:
                    best = exp
                    best_diff = diff
        return best

    def _avg_chain_iv(self, chain: list[OptionQuote]) -> float:
        ivs = [q.greeks.iv for q in chain if q.greeks.iv > 0]
        return sum(ivs) / len(ivs) if ivs else 0.0

    def _build_iron_condor(
        self,
        symbol: str,
        spot: float,
        expiration: date,
        chain: list[OptionQuote],
    ) -> list[Order]:
        calls = sorted(
            [q for q in chain if q.contract.option_type == OptionType.CALL and q.bid > 0],
            key=lambda q: q.contract.strike,
        )
        puts = sorted(
            [q for q in chain if q.contract.option_type == OptionType.PUT and q.bid > 0],
            key=lambda q: q.contract.strike,
        )

        # Find short strikes by delta
        short_call = self._find_by_delta(calls, self.short_delta)
        short_put = self._find_by_delta(puts, -self.short_delta)
        if not short_call or not short_put:
            return []

        # Wing strikes
        long_call = self._find_by_strike(calls, short_call.contract.strike + self.wing_width)
        long_put = self._find_by_strike(puts, short_put.contract.strike - self.wing_width)
        if not long_call or not long_put:
            return []

        # Calculate credit
        credit = (short_call.bid + short_put.bid) - (long_call.ask + long_put.ask)
        if credit < self.min_credit:
            log.debug("condor_credit_too_low", credit=credit, min=self.min_credit)
            return []

        # Position sizing
        max_loss_per = (self.wing_width - credit) * 100
        quantity = max(1, int(self.max_risk_per_trade / max_loss_per)) if max_loss_per > 0 else 1

        log.info(
            "iron_condor",
            symbol=symbol,
            put_wing=long_put.contract.strike,
            short_put=short_put.contract.strike,
            short_call=short_call.contract.strike,
            call_wing=long_call.contract.strike,
            credit=credit,
            qty=quantity,
        )

        return [
            Order(contract=long_put.contract, side=Side.BUY, quantity=quantity,
                  order_type=OrderType.LIMIT, limit_price=long_put.ask),
            Order(contract=short_put.contract, side=Side.SELL, quantity=quantity,
                  order_type=OrderType.LIMIT, limit_price=short_put.bid),
            Order(contract=short_call.contract, side=Side.SELL, quantity=quantity,
                  order_type=OrderType.LIMIT, limit_price=short_call.bid),
            Order(contract=long_call.contract, side=Side.BUY, quantity=quantity,
                  order_type=OrderType.LIMIT, limit_price=long_call.ask),
        ]

    def _find_by_delta(self, quotes: list[OptionQuote], target_delta: float) -> OptionQuote | None:
        valid = [q for q in quotes if q.greeks.delta != 0]
        if not valid:
            return None
        return min(valid, key=lambda q: abs(q.greeks.delta - target_delta))

    def _find_by_strike(self, quotes: list[OptionQuote], target_strike: float) -> OptionQuote | None:
        if not quotes:
            return None
        return min(quotes, key=lambda q: abs(q.contract.strike - target_strike))

    async def on_fill(self, order: Order) -> None:
        log.info("condor_fill", contract=str(order.contract), side=order.side.value, price=order.fill_price)
