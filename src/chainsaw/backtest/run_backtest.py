"""Standalone backtest runner using synthetic market data.

Run: python -m chainsaw.backtest.run_backtest

Uses historical price patterns to generate realistic test data
without requiring any external API or broker connection.
"""

from __future__ import annotations

import sys
import os

# Add src to path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from datetime import date, timedelta

from chainsaw.backtest.engine import BacktestConfig, BacktestEngine
from chainsaw.models import (
    OptionContract,
    OptionQuote,
    OptionType,
    Order,
    OrderType,
    PortfolioSnapshot,
    Side,
    Greeks,
    StockQuote,
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
    """Generate realistic daily OHLCV data using geometric Brownian motion."""
    np.random.seed(seed)
    trading_days = pd.bdate_range(start, end)
    n = len(trading_days)

    daily_return = annual_return / 252
    daily_vol = annual_vol / np.sqrt(252)

    # GBM returns
    returns = np.random.normal(daily_return, daily_vol, n)
    prices = initial_price * np.cumprod(1 + returns)

    # Generate OHLCV from close prices
    intraday_vol = daily_vol * 0.6
    highs = prices * (1 + np.abs(np.random.normal(0, intraday_vol, n)))
    lows = prices * (1 - np.abs(np.random.normal(0, intraday_vol, n)))
    opens = np.roll(prices, 1) * (1 + np.random.normal(0, daily_vol * 0.3, n))
    opens[0] = initial_price
    volume = np.random.randint(50_000_000, 150_000_000, n)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volume,
    }, index=trading_days)

    # Ensure high >= max(open, close) and low <= min(open, close)
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
    """Generate synthetic IV data that correlates with price moves."""
    np.random.seed(seed)
    n = len(price_data)
    returns = price_data["close"].pct_change().fillna(0)

    iv = np.zeros(n)
    iv[0] = base_iv

    for i in range(1, n):
        # IV tends to spike on down moves and revert to mean
        shock = np.random.normal(0, vol_of_vol)
        reversion = mean_reversion * (base_iv - iv[i - 1])
        leverage_effect = -0.5 * returns.iloc[i]  # Negative correlation
        iv[i] = max(0.05, iv[i - 1] + reversion + leverage_effect + shock)

    return pd.DataFrame({"iv": iv}, index=price_data.index)


# ---------------------------------------------------------------------------
# Backtest-compatible strategies (don't need real broker/data providers)
# ---------------------------------------------------------------------------

class BacktestVerticalSpread(Strategy):
    """Simplified vertical spread for backtesting against synthetic data."""

    def __init__(
        self,
        pricing: PricingEngine,
        signal_gen: CompositeSignalGenerator,
        spread_width: float = 5.0,
        target_dte: int = 30,
        max_risk: float = 500.0,
        min_signal_strength: float = 0.25,
        max_open_positions: int = 5,
        cooldown_days: int = 5,
    ) -> None:
        super().__init__(name="vertical_spread_bt")
        self.pricing = pricing
        self.signals = signal_gen
        self.spread_width = spread_width
        self.target_dte = target_dte
        self.max_risk = max_risk
        self.min_strength = min_signal_strength
        self.max_open = max_open_positions
        self.cooldown = cooldown_days
        self._price_history: list[float] = []
        self._iv_history: list[float] = []
        self._days_since_trade: int = 999
        self._position_open = False

    async def evaluate(self, portfolio: PortfolioSnapshot) -> Signal:
        # Track price history from portfolio value as proxy
        self._days_since_trade += 1

        # Allow multiple concurrent positions with cooldown
        if len(portfolio.positions) >= self.max_open * 2:  # *2 because each spread = 2 legs
            return Signal(signal_type=SignalType.HOLD, reason="Max positions reached")

        if self._days_since_trade < self.cooldown:
            return Signal(signal_type=SignalType.HOLD, reason="Cooldown period")

        if len(self._price_history) < 35:
            return Signal(signal_type=SignalType.HOLD, reason="Building price history")

        prices = pd.Series(self._price_history)
        iv_series = pd.Series(self._iv_history) if self._iv_history else None
        current_iv = self._iv_history[-1] if self._iv_history else 0.20

        output = self.signals.generate(prices, current_iv=current_iv, iv_history=iv_series)

        if output.direction == "neutral" or output.strength < self.min_strength:
            return Signal(signal_type=SignalType.HOLD, reason=output.rationale)

        spot = self._price_history[-1]
        expiration = date.today() + timedelta(days=self.target_dte)

        if output.direction == "bull":
            long_strike = round(spot / 5) * 5  # Round to nearest 5
            short_strike = long_strike + self.spread_width
            long_contract = OptionContract(symbol="SPY", expiration=expiration,
                                           strike=long_strike, option_type=OptionType.CALL)
            short_contract = OptionContract(symbol="SPY", expiration=expiration,
                                            strike=short_strike, option_type=OptionType.CALL)
        else:
            long_strike = round(spot / 5) * 5
            short_strike = long_strike - self.spread_width
            long_contract = OptionContract(symbol="SPY", expiration=expiration,
                                           strike=long_strike, option_type=OptionType.PUT)
            short_contract = OptionContract(symbol="SPY", expiration=expiration,
                                            strike=short_strike, option_type=OptionType.PUT)

        # Price the legs
        long_price = self.pricing.price(long_contract, spot, current_iv)
        short_price = self.pricing.price(short_contract, spot, current_iv)
        spread_cost = long_price.theoretical_price - short_price.theoretical_price

        if spread_cost <= 0.10:
            return Signal(signal_type=SignalType.HOLD, reason="Spread too cheap")

        qty = max(1, min(int(self.max_risk / (spread_cost * 100)), 5))

        orders = [
            Order(contract=long_contract, side=Side.BUY, quantity=qty,
                  order_type=OrderType.LIMIT, limit_price=long_price.theoretical_price),
            Order(contract=short_contract, side=Side.SELL, quantity=qty,
                  order_type=OrderType.LIMIT, limit_price=short_price.theoretical_price),
        ]

        self._days_since_trade = 0
        return Signal(
            signal_type=SignalType.OPEN,
            orders=orders,
            reason=f"{'Bull call' if output.direction == 'bull' else 'Bear put'} spread ({output.rationale})",
            confidence=output.strength,
        )

    async def on_fill(self, order: Order) -> None:
        pass

    def feed_price(self, price: float, iv: float = 0.20) -> None:
        """Feed daily price data to build signal history."""
        self._price_history.append(price)
        self._iv_history.append(iv)


class BacktestIronCondor(Strategy):
    """Simplified iron condor for backtesting."""

    def __init__(
        self,
        pricing: PricingEngine,
        signal_gen: CompositeSignalGenerator,
        wing_width: float = 5.0,
        short_delta_target: float = 0.20,
        target_dte: int = 30,
        max_risk: float = 500.0,
        min_iv_rank: float = 30.0,
        max_open_positions: int = 5,
        cooldown_days: int = 5,
    ) -> None:
        super().__init__(name="iron_condor_bt")
        self.pricing = pricing
        self.signals = signal_gen
        self.wing_width = wing_width
        self.short_delta_target = short_delta_target
        self.target_dte = target_dte
        self.max_risk = max_risk
        self.min_iv_rank = min_iv_rank
        self.max_open = max_open_positions
        self.cooldown = cooldown_days
        self._price_history: list[float] = []
        self._iv_history: list[float] = []
        self._days_since_trade: int = 999

    async def evaluate(self, portfolio: PortfolioSnapshot) -> Signal:
        self._days_since_trade += 1

        if len(portfolio.positions) >= self.max_open * 4:  # 4 legs per condor
            return Signal(signal_type=SignalType.HOLD, reason="Max condors open")

        if self._days_since_trade < self.cooldown:
            return Signal(signal_type=SignalType.HOLD, reason="Cooldown period")

        if len(self._iv_history) < 20:
            return Signal(signal_type=SignalType.HOLD, reason="Building IV history")

        iv_series = pd.Series(self._iv_history)
        current_iv = self._iv_history[-1]

        if not self.signals.should_sell_premium(current_iv, iv_series):
            return Signal(signal_type=SignalType.HOLD, reason="IV rank too low for selling")

        spot = self._price_history[-1]
        expiration = date.today() + timedelta(days=self.target_dte)

        # Find strikes roughly at target delta distance
        # Approximate: delta ~0.20 is about 1 standard deviation OTM
        std_move = spot * current_iv * np.sqrt(self.target_dte / 365)
        short_call_strike = round((spot + std_move) / 5) * 5
        short_put_strike = round((spot - std_move) / 5) * 5
        long_call_strike = short_call_strike + self.wing_width
        long_put_strike = short_put_strike - self.wing_width

        contracts = {
            "long_put": OptionContract("SPY", expiration, long_put_strike, OptionType.PUT),
            "short_put": OptionContract("SPY", expiration, short_put_strike, OptionType.PUT),
            "short_call": OptionContract("SPY", expiration, short_call_strike, OptionType.CALL),
            "long_call": OptionContract("SPY", expiration, long_call_strike, OptionType.CALL),
        }

        prices = {k: self.pricing.price(c, spot, current_iv) for k, c in contracts.items()}
        credit = (prices["short_put"].theoretical_price + prices["short_call"].theoretical_price -
                  prices["long_put"].theoretical_price - prices["long_call"].theoretical_price)

        if credit < 0.30:
            return Signal(signal_type=SignalType.HOLD, reason="Credit too small")

        max_loss = (self.wing_width - credit) * 100
        qty = max(1, int(self.max_risk / max_loss)) if max_loss > 0 else 1

        orders = [
            Order(contract=contracts["long_put"], side=Side.BUY, quantity=qty,
                  order_type=OrderType.LIMIT, limit_price=prices["long_put"].theoretical_price),
            Order(contract=contracts["short_put"], side=Side.SELL, quantity=qty,
                  order_type=OrderType.LIMIT, limit_price=prices["short_put"].theoretical_price),
            Order(contract=contracts["short_call"], side=Side.SELL, quantity=qty,
                  order_type=OrderType.LIMIT, limit_price=prices["short_call"].theoretical_price),
            Order(contract=contracts["long_call"], side=Side.BUY, quantity=qty,
                  order_type=OrderType.LIMIT, limit_price=prices["long_call"].theoretical_price),
        ]

        self._days_since_trade = 0
        return Signal(
            signal_type=SignalType.OPEN,
            orders=orders,
            reason=f"Iron condor (IV elevated)",
            confidence=0.5,
        )

    async def on_fill(self, order: Order) -> None:
        pass

    def feed_price(self, price: float, iv: float = 0.20) -> None:
        self._price_history.append(price)
        self._iv_history.append(iv)


# ---------------------------------------------------------------------------
# Custom backtest engine that feeds price data to strategies
# ---------------------------------------------------------------------------

class DataFeedingBacktestEngine(BacktestEngine):
    """Extended backtest engine that feeds prices to strategies each day."""

    def run(
        self,
        strategy: Strategy,
        price_data: pd.DataFrame,
        iv_data: pd.DataFrame | None = None,
    ):
        import asyncio

        dates = self._get_trading_dates(price_data)
        log.info("backtest_start", strategy=strategy.name, dates=len(dates))

        for current_date in dates:
            row = price_data.loc[current_date]
            spot = float(row["close"])
            iv = float(iv_data.loc[current_date]["iv"]) if iv_data is not None and current_date in iv_data.index else 0.20

            # Feed price to strategy if it supports it
            if hasattr(strategy, "feed_price"):
                strategy.feed_price(spot, iv)

            # Update position prices
            self._mark_positions(current_date, spot, iv)

            # Expire worthless options
            self._handle_expirations(current_date, spot)

            # Build simulated portfolio snapshot
            portfolio = self._build_portfolio(current_date)

            # Evaluate strategy
            try:
                loop = asyncio.new_event_loop()
                signal = loop.run_until_complete(strategy.evaluate(portfolio))
                loop.close()
            except Exception as e:
                log.debug("eval_error", error=str(e))
                signal = Signal(signal_type=SignalType.HOLD)

            # Process signal orders
            if signal.signal_type != SignalType.HOLD:
                self._process_signal(signal, current_date, spot, iv)

            # Record equity
            equity = self._calculate_equity(spot)
            self._peak_equity = max(self._peak_equity, equity)
            drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0
            self._equity_history.append({
                "date": current_date,
                "equity": equity,
                "drawdown": drawdown,
                "cash": self._cash,
                "positions_value": equity - self._cash,
            })

        # Close remaining positions at final prices
        if dates:
            final_row = price_data.loc[dates[-1]]
            self._close_all_positions(dates[-1], float(final_row["close"]))

        return self._compile_results()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backtest():
    print("=" * 70)
    print("  LEGENDARY CHAINSAW — Options Strategy Backtest")
    print("=" * 70)
    print()

    # Configuration
    start = date(2024, 1, 2)
    end = date(2025, 12, 31)

    print(f"  Period:  {start} → {end}")
    print(f"  Capital: $100,000")
    print()

    # Generate synthetic market data for multiple scenarios
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
    signal_gen = CompositeSignalGenerator(
        bull_threshold=0.10,
        bear_threshold=-0.10,
        min_iv_rank_for_selling=20.0,
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
            price_data,
            base_iv=params["base_iv"],
            seed=params["seed"] + 1,
        )

        print(f"  Price range: ${price_data['close'].min():.2f} — ${price_data['close'].max():.2f}")
        print(f"  IV range:    {iv_data['iv'].min():.1%} — {iv_data['iv'].max():.1%}")
        print()

        # --- Vertical Spread Backtest ---
        config = BacktestConfig(start_date=start, end_date=end, initial_capital=100_000)
        engine = DataFeedingBacktestEngine(config, pricing)
        strategy = BacktestVerticalSpread(pricing, signal_gen, min_signal_strength=0.10, target_dte=21)
        result = engine.run(strategy, price_data, iv_data)

        print("  📊 VERTICAL SPREAD STRATEGY")
        print(f"    Total Return:      {result.total_return:>8.2%}")
        print(f"    Annualized Return: {result.annualized_return:>8.2%}")
        print(f"    Max Drawdown:      {result.max_drawdown:>8.2%}")
        print(f"    Sharpe Ratio:      {result.sharpe_ratio:>8.2f}")
        print(f"    Win Rate:          {result.win_rate:>8.2%}")
        print(f"    Profit Factor:     {result.profit_factor:>8.2f}")
        print(f"    Total Trades:      {result.total_trades:>8d}")
        print(f"    Avg Trade P&L:     ${result.avg_trade_pnl:>7.2f}")
        print()

        # --- Iron Condor Backtest ---
        config2 = BacktestConfig(start_date=start, end_date=end, initial_capital=100_000)
        engine2 = DataFeedingBacktestEngine(config2, pricing)
        strategy2 = BacktestIronCondor(pricing, signal_gen, min_iv_rank=20.0, target_dte=21)
        result2 = engine2.run(strategy2, price_data, iv_data)

        print("  📊 IRON CONDOR STRATEGY")
        print(f"    Total Return:      {result2.total_return:>8.2%}")
        print(f"    Annualized Return: {result2.annualized_return:>8.2%}")
        print(f"    Max Drawdown:      {result2.max_drawdown:>8.2%}")
        print(f"    Sharpe Ratio:      {result2.sharpe_ratio:>8.2f}")
        print(f"    Win Rate:          {result2.win_rate:>8.2%}")
        print(f"    Profit Factor:     {result2.profit_factor:>8.2f}")
        print(f"    Total Trades:      {result2.total_trades:>8d}")
        print(f"    Avg Trade P&L:     ${result2.avg_trade_pnl:>7.2f}")
        print()

    print("=" * 70)
    print("  Backtest complete. These results use synthetic data with realistic")
    print("  statistical properties (GBM + stochastic vol). Live results will")
    print("  differ — use as directional guidance, not prediction.")
    print("=" * 70)


if __name__ == "__main__":
    run_backtest()
