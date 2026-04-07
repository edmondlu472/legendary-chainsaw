"""Standalone backtest runner with improved exit management and position sizing.

Run: python -m chainsaw.backtest.run_backtest
"""

from __future__ import annotations

import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from datetime import date, timedelta

from chainsaw.backtest.engine import BacktestConfig, BacktestEngine, SpreadPosition, LegPosition
from chainsaw.models import (
    OptionContract,
    OptionType,
    Order,
    OrderType,
    PortfolioSnapshot,
    Side,
)
from chainsaw.pricing.engine import PricingEngine
from chainsaw.signals.composite import CompositeSignalGenerator
from chainsaw.strategy.base import Signal, SignalType, Strategy
from chainsaw.logging import setup_logging, get_logger

setup_logging("INFO")
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def generate_stock_data(
    symbol: str,
    start: date,
    end: date,
    initial_price: float = 450.0,
    annual_return: float = 0.10,
    annual_vol: float = 0.18,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    trading_days = pd.bdate_range(start, end)
    n = len(trading_days)

    daily_return = annual_return / 252
    daily_vol = annual_vol / np.sqrt(252)

    returns = np.random.normal(daily_return, daily_vol, n)
    prices = initial_price * np.cumprod(1 + returns)

    intraday_vol = daily_vol * 0.6
    highs = prices * (1 + np.abs(np.random.normal(0, intraday_vol, n)))
    lows = prices * (1 - np.abs(np.random.normal(0, intraday_vol, n)))
    opens = np.roll(prices, 1) * (1 + np.random.normal(0, daily_vol * 0.3, n))
    opens[0] = initial_price
    volume = np.random.randint(50_000_000, 150_000_000, n)

    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": prices, "volume": volume,
    }, index=trading_days)

    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def generate_iv_data(
    price_data: pd.DataFrame,
    base_iv: float = 0.20,
    mean_reversion: float = 0.05,
    vol_of_vol: float = 0.03,
    seed: int = 123,
) -> pd.DataFrame:
    np.random.seed(seed)
    n = len(price_data)
    returns = price_data["close"].pct_change().fillna(0)

    iv = np.zeros(n)
    iv[0] = base_iv

    for i in range(1, n):
        shock = np.random.normal(0, vol_of_vol)
        reversion = mean_reversion * (base_iv - iv[i - 1])
        leverage_effect = -0.5 * returns.iloc[i]
        iv[i] = max(0.05, iv[i - 1] + reversion + leverage_effect + shock)

    return pd.DataFrame({"iv": iv}, index=price_data.index)


# ---------------------------------------------------------------------------
# Improved strategies with proper exit management
# ---------------------------------------------------------------------------

class ImprovedVerticalSpread(Strategy):
    """Vertical spread with:
    - Take profit at 50% of max gain
    - Stop loss at 100% of entry cost (1:1 risk/reward)
    - Close at 5 DTE to avoid gamma risk
    - Dynamic sizing based on signal strength (Kelly-inspired)
    - Momentum-first signal weighting
    """

    def __init__(
        self,
        pricing: PricingEngine,
        signal_gen: CompositeSignalGenerator,
        spread_width: float = 5.0,
        target_dte: int = 30,
        base_risk_pct: float = 0.02,  # 2% of portfolio per trade
        min_signal_strength: float = 0.08,
        max_open_spreads: int = 8,
        cooldown_days: int = 3,
    ) -> None:
        super().__init__(name="vertical_spread_v2")
        self.pricing = pricing
        self.signals = signal_gen
        self.spread_width = spread_width
        self.target_dte = target_dte
        self.base_risk_pct = base_risk_pct
        self.min_strength = min_signal_strength
        self.max_open = max_open_spreads
        self.cooldown = cooldown_days
        self._price_history: list[float] = []
        self._iv_history: list[float] = []
        self._days_since_trade: int = 999

    async def evaluate(self, portfolio: PortfolioSnapshot) -> Signal:
        self._days_since_trade += 1

        # Count open spreads (2 legs each)
        open_spreads = len(portfolio.positions) // 2
        if open_spreads >= self.max_open:
            return Signal(signal_type=SignalType.HOLD, reason="Max spreads open")

        if self._days_since_trade < self.cooldown:
            return Signal(signal_type=SignalType.HOLD, reason="Cooldown")

        if len(self._price_history) < 40:
            return Signal(signal_type=SignalType.HOLD, reason="Building history")

        prices = pd.Series(self._price_history)
        iv_series = pd.Series(self._iv_history) if self._iv_history else None
        current_iv = self._iv_history[-1] if self._iv_history else 0.20

        output = self.signals.generate(prices, current_iv=current_iv, iv_history=iv_series)

        if output.direction == "neutral" or output.strength < self.min_strength:
            return Signal(signal_type=SignalType.HOLD, reason=output.rationale)

        spot = self._price_history[-1]
        expiration = date.today() + timedelta(days=self.target_dte)

        if output.direction == "bull":
            # Slightly ITM long strike for higher delta exposure
            long_strike = round(spot / 5) * 5
            short_strike = long_strike + self.spread_width
            long_contract = OptionContract("SPY", expiration, long_strike, OptionType.CALL)
            short_contract = OptionContract("SPY", expiration, short_strike, OptionType.CALL)
        else:
            long_strike = round(spot / 5) * 5
            short_strike = long_strike - self.spread_width
            long_contract = OptionContract("SPY", expiration, long_strike, OptionType.PUT)
            short_contract = OptionContract("SPY", expiration, short_strike, OptionType.PUT)

        long_price = self.pricing.price(long_contract, spot, current_iv)
        short_price = self.pricing.price(short_contract, spot, current_iv)
        spread_cost = long_price.theoretical_price - short_price.theoretical_price

        if spread_cost <= 0.10:
            return Signal(signal_type=SignalType.HOLD, reason="Spread too cheap")

        # Dynamic position sizing: scale with signal strength
        # Kelly-inspired: size = edge * bankroll / odds
        risk_per_trade = portfolio.net_liquidation * self.base_risk_pct * min(output.strength * 3, 1.5)
        qty = max(1, min(int(risk_per_trade / (spread_cost * 100)), 10))

        orders = [
            Order(contract=long_contract, side=Side.BUY, quantity=qty,
                  order_type=OrderType.LIMIT, limit_price=long_price.theoretical_price),
            Order(contract=short_contract, side=Side.SELL, quantity=qty,
                  order_type=OrderType.LIMIT, limit_price=short_price.theoretical_price),
        ]

        self._days_since_trade = 0
        signal = Signal(
            signal_type=SignalType.OPEN,
            orders=orders,
            reason=f"{'Bull call' if output.direction == 'bull' else 'Bear put'} spread ({output.rationale})",
            confidence=output.strength,
        )
        # Attach exit params for the engine to use
        signal._take_profit_pct = 0.50
        signal._stop_loss_pct = 1.0  # 1:1 risk/reward stop
        return signal

    async def on_fill(self, order: Order) -> None:
        pass

    def feed_price(self, price: float, iv: float = 0.20) -> None:
        self._price_history.append(price)
        self._iv_history.append(iv)


class ImprovedIronCondor(Strategy):
    """Iron condor with:
    - Wider wings (10 wide instead of 5)
    - Farther OTM short strikes (1.2 std devs)
    - Take profit at 50% of credit received
    - Stop loss at 2x credit (i.e., spread value doubles)
    - Only enter when IV rank > 40
    - Close at 7 DTE
    - Dynamic sizing
    """

    def __init__(
        self,
        pricing: PricingEngine,
        signal_gen: CompositeSignalGenerator,
        wing_width: float = 10.0,
        short_strike_std: float = 1.2,
        target_dte: int = 35,
        base_risk_pct: float = 0.02,
        min_iv_rank: float = 40.0,
        max_open_condors: int = 6,
        cooldown_days: int = 5,
    ) -> None:
        super().__init__(name="iron_condor_v2")
        self.pricing = pricing
        self.signals = signal_gen
        self.wing_width = wing_width
        self.short_strike_std = short_strike_std
        self.target_dte = target_dte
        self.base_risk_pct = base_risk_pct
        self.min_iv_rank = min_iv_rank
        self.max_open = max_open_condors
        self.cooldown = cooldown_days
        self._price_history: list[float] = []
        self._iv_history: list[float] = []
        self._days_since_trade: int = 999

    async def evaluate(self, portfolio: PortfolioSnapshot) -> Signal:
        self._days_since_trade += 1

        open_condors = len(portfolio.positions) // 4
        if open_condors >= self.max_open:
            return Signal(signal_type=SignalType.HOLD, reason="Max condors open")

        if self._days_since_trade < self.cooldown:
            return Signal(signal_type=SignalType.HOLD, reason="Cooldown")

        if len(self._iv_history) < 30:
            return Signal(signal_type=SignalType.HOLD, reason="Building history")

        iv_series = pd.Series(self._iv_history)
        current_iv = self._iv_history[-1]

        if not self.signals.should_sell_premium(current_iv, iv_series):
            return Signal(signal_type=SignalType.HOLD, reason="IV rank too low")

        # Also check that price is range-bound (low momentum = good for condors)
        prices = pd.Series(self._price_history)
        output = self.signals.generate(prices, current_iv=current_iv, iv_history=iv_series)
        if output.strength > 0.5:
            # Strong directional signal = bad for condors
            return Signal(signal_type=SignalType.HOLD, reason="Too directional for condor")

        spot = self._price_history[-1]
        expiration = date.today() + timedelta(days=self.target_dte)

        # Place short strikes at ~1.2 standard deviations
        std_move = spot * current_iv * np.sqrt(self.target_dte / 365) * self.short_strike_std
        short_call_strike = round((spot + std_move) / 5) * 5
        short_put_strike = round((spot - std_move) / 5) * 5

        # Ensure short strikes are at least $10 apart (avoid narrow condors)
        if short_call_strike - short_put_strike < 10:
            short_call_strike = round(spot / 5) * 5 + 5
            short_put_strike = round(spot / 5) * 5 - 5

        long_call_strike = short_call_strike + self.wing_width
        long_put_strike = short_put_strike - self.wing_width

        contracts = {
            "long_put": OptionContract("SPY", expiration, long_put_strike, OptionType.PUT),
            "short_put": OptionContract("SPY", expiration, short_put_strike, OptionType.PUT),
            "short_call": OptionContract("SPY", expiration, short_call_strike, OptionType.CALL),
            "long_call": OptionContract("SPY", expiration, long_call_strike, OptionType.CALL),
        }

        prices_dict = {k: self.pricing.price(c, spot, current_iv) for k, c in contracts.items()}
        credit = (prices_dict["short_put"].theoretical_price + prices_dict["short_call"].theoretical_price -
                  prices_dict["long_put"].theoretical_price - prices_dict["long_call"].theoretical_price)

        if credit < 0.50:
            return Signal(signal_type=SignalType.HOLD, reason=f"Credit too small ({credit:.2f})")

        # Position sizing based on max loss
        max_loss_per_contract = (self.wing_width - credit) * 100
        risk_budget = portfolio.net_liquidation * self.base_risk_pct
        qty = max(1, min(int(risk_budget / max_loss_per_contract), 8))

        orders = [
            Order(contract=contracts["long_put"], side=Side.BUY, quantity=qty,
                  order_type=OrderType.LIMIT, limit_price=prices_dict["long_put"].theoretical_price),
            Order(contract=contracts["short_put"], side=Side.SELL, quantity=qty,
                  order_type=OrderType.LIMIT, limit_price=prices_dict["short_put"].theoretical_price),
            Order(contract=contracts["short_call"], side=Side.SELL, quantity=qty,
                  order_type=OrderType.LIMIT, limit_price=prices_dict["short_call"].theoretical_price),
            Order(contract=contracts["long_call"], side=Side.BUY, quantity=qty,
                  order_type=OrderType.LIMIT, limit_price=prices_dict["long_call"].theoretical_price),
        ]

        self._days_since_trade = 0
        signal = Signal(
            signal_type=SignalType.OPEN,
            orders=orders,
            reason=f"Iron condor ${short_put_strike}p/${short_call_strike}c (IV rank elevated)",
            confidence=0.5,
        )
        signal._take_profit_pct = 0.50  # Close at 50% of credit
        signal._stop_loss_pct = 1.5     # Stop at 1.5x max loss
        return signal

    async def on_fill(self, order: Order) -> None:
        pass

    def feed_price(self, price: float, iv: float = 0.20) -> None:
        self._price_history.append(price)
        self._iv_history.append(iv)


# ---------------------------------------------------------------------------
# Extended backtest engine with data feeding
# ---------------------------------------------------------------------------

class DataFeedingBacktestEngine(BacktestEngine):
    def run(self, strategy: Strategy, price_data: pd.DataFrame, iv_data: pd.DataFrame | None = None):
        dates = self._get_trading_dates(price_data)
        log.info("backtest_start", strategy=strategy.name, dates=len(dates))

        for current_date in dates:
            row = price_data.loc[current_date]
            spot = float(row["close"])
            iv = float(iv_data.loc[current_date]["iv"]) if iv_data is not None and current_date in iv_data.index else 0.20

            if hasattr(strategy, "feed_price"):
                strategy.feed_price(spot, iv)

            # Update all position prices
            self._mark_positions(current_date, spot, iv)

            # Check exits BEFORE evaluating new entries
            self._check_exits(current_date, spot)

            # Build portfolio and evaluate
            portfolio = self._build_portfolio(current_date)

            try:
                loop = asyncio.new_event_loop()
                signal = loop.run_until_complete(strategy.evaluate(portfolio))
                loop.close()
            except Exception as e:
                log.debug("eval_error", error=str(e))
                signal = Signal(signal_type=SignalType.HOLD)

            if signal.signal_type != SignalType.HOLD:
                self._process_signal(signal, current_date, spot, iv)

            equity = self._calculate_equity()
            self._peak_equity = max(self._peak_equity, equity)
            drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0
            self._equity_history.append({
                "date": current_date,
                "equity": equity,
                "drawdown": drawdown,
                "cash": self._cash,
                "positions_value": equity - self._cash,
            })

        if dates:
            self._close_all_positions(dates[-1], float(price_data.loc[dates[-1]]["close"]))

        return self._compile_results()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backtest():
    print("=" * 70)
    print("  LEGENDARY CHAINSAW — Options Strategy Backtest v2")
    print("  (with exit management, dynamic sizing, improved entries)")
    print("=" * 70)
    print()

    start = date(2024, 1, 2)
    end = date(2025, 12, 31)

    print(f"  Period:  {start} → {end}")
    print(f"  Capital: $100,000")
    print(f"  Exits:   TP @ 50% max profit | SL @ 1-1.5x | Close <5 DTE")
    print(f"  Sizing:  2% risk per trade, scaled by signal strength")
    print()

    scenarios = {
        "Bull Market (SPY +15%, Vol 16%)": {
            "annual_return": 0.15, "annual_vol": 0.16, "base_iv": 0.18, "seed": 42,
        },
        "Choppy Market (SPY +3%, Vol 22%)": {
            "annual_return": 0.03, "annual_vol": 0.22, "base_iv": 0.24, "seed": 99,
        },
        "Bear Market (SPY -12%, Vol 28%)": {
            "annual_return": -0.12, "annual_vol": 0.28, "base_iv": 0.30, "seed": 77,
        },
    }

    pricing = PricingEngine(risk_free_rate=0.05)

    # Momentum-heavy signal gen for directional spreads
    spread_signals = CompositeSignalGenerator(
        momentum_weight=0.7,
        mean_reversion_weight=0.2,
        iv_weight=0.1,
        bull_threshold=0.08,
        bear_threshold=-0.08,
    )

    # Neutral-biased signal gen for condors (high IV rank filter)
    condor_signals = CompositeSignalGenerator(
        momentum_weight=0.3,
        mean_reversion_weight=0.3,
        iv_weight=0.4,
        min_iv_rank_for_selling=40.0,
    )

    for scenario_name, params in scenarios.items():
        print("-" * 70)
        print(f"  SCENARIO: {scenario_name}")
        print("-" * 70)

        price_data = generate_stock_data(
            "SPY", start, end,
            initial_price=450.0,
            annual_return=params["annual_return"],
            annual_vol=params["annual_vol"],
            seed=params["seed"],
        )
        iv_data = generate_iv_data(
            price_data, base_iv=params["base_iv"], seed=params["seed"] + 1,
        )

        print(f"  Price range: ${price_data['close'].min():.2f} — ${price_data['close'].max():.2f}")
        print(f"  IV range:    {iv_data['iv'].min():.1%} — {iv_data['iv'].max():.1%}")
        print()

        # --- Vertical Spread ---
        config = BacktestConfig(start_date=start, end_date=end, initial_capital=100_000)
        engine = DataFeedingBacktestEngine(config, pricing)
        strategy = ImprovedVerticalSpread(pricing, spread_signals)
        result = engine.run(strategy, price_data, iv_data)

        _print_result("VERTICAL SPREAD v2", result)

        # --- Iron Condor ---
        config2 = BacktestConfig(start_date=start, end_date=end, initial_capital=100_000)
        engine2 = DataFeedingBacktestEngine(config2, pricing)
        strategy2 = ImprovedIronCondor(pricing, condor_signals)
        result2 = engine2.run(strategy2, price_data, iv_data)

        _print_result("IRON CONDOR v2", result2)

        # --- Combined Strategy ---
        config3 = BacktestConfig(start_date=start, end_date=end, initial_capital=100_000)
        engine3 = DataFeedingBacktestEngine(config3, pricing)
        # Create both strategies sharing the same feed
        spread_strat = ImprovedVerticalSpread(pricing, spread_signals, base_risk_pct=0.01)
        condor_strat = ImprovedIronCondor(pricing, condor_signals, base_risk_pct=0.01)
        combined = CombinedStrategy(spread_strat, condor_strat)
        result3 = engine3.run(combined, price_data, iv_data)

        _print_result("COMBINED (50/50 allocation)", result3)

    print("=" * 70)
    print("  Key improvements over v1:")
    print("    - Take-profit exits lock in gains at 50% of max")
    print("    - Stop-losses prevent catastrophic single-trade losses")
    print("    - Gamma risk exit closes positions before expiration")
    print("    - Dynamic sizing scales with signal conviction")
    print("    - Wider condor wings = more room for price movement")
    print("    - IV rank filtering only sells premium in high-IV regimes")
    print("=" * 70)


class CombinedStrategy(Strategy):
    """Runs both strategies with shared position limits."""

    def __init__(self, spread: ImprovedVerticalSpread, condor: ImprovedIronCondor) -> None:
        super().__init__(name="combined_v2")
        self.spread = spread
        self.condor = condor

    async def evaluate(self, portfolio: PortfolioSnapshot) -> Signal:
        # Try spread first (higher priority when directional)
        signal = await self.spread.evaluate(portfolio)
        if signal.signal_type != SignalType.HOLD:
            return signal

        # Then try condor
        return await self.condor.evaluate(portfolio)

    async def on_fill(self, order: Order) -> None:
        pass

    def feed_price(self, price: float, iv: float = 0.20) -> None:
        self.spread.feed_price(price, iv)
        self.condor.feed_price(price, iv)


def _print_result(name: str, result) -> None:
    # Count exit reasons
    exit_counts: dict[str, int] = {}
    for t in result.trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    print(f"  {name}")
    print(f"    Total Return:      {result.total_return:>8.2%}")
    print(f"    Annualized Return: {result.annualized_return:>8.2%}")
    print(f"    Max Drawdown:      {result.max_drawdown:>8.2%}")
    print(f"    Sharpe Ratio:      {result.sharpe_ratio:>8.2f}")
    print(f"    Win Rate:          {result.win_rate:>8.2%}")
    print(f"    Profit Factor:     {result.profit_factor:>8.2f}")
    print(f"    Total Trades:      {result.total_trades:>8d}")
    print(f"    Avg Trade P&L:     ${result.avg_trade_pnl:>7.2f}")
    if exit_counts:
        exits_str = ", ".join(f"{k}: {v}" for k, v in sorted(exit_counts.items()))
        print(f"    Exit Reasons:      {exits_str}")
    print()


if __name__ == "__main__":
    run_backtest()
