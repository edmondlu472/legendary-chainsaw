"""Standalone backtest runner v3 — optimized for Sharpe ratio and win rate.

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

from chainsaw.backtest.engine import BacktestConfig, BacktestEngine
from chainsaw.models import (
    OptionContract, OptionType, Order, OrderType, PortfolioSnapshot, Side,
)
from chainsaw.pricing.engine import PricingEngine
from chainsaw.signals.indicators import ema, sma, rsi
from chainsaw.strategy.base import Signal, SignalType, Strategy
from chainsaw.logging import setup_logging, get_logger

setup_logging("WARNING")
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_stock_data(
    symbol: str, start: date, end: date, initial_price: float = 450.0,
    annual_return: float = 0.10, annual_vol: float = 0.18, seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    days = pd.bdate_range(start, end)
    n = len(days)
    dr, dv = annual_return / 252, annual_vol / np.sqrt(252)
    rets = np.random.normal(dr, dv, n)
    close = initial_price * np.cumprod(1 + rets)
    iv = dv * 0.6
    high = close * (1 + np.abs(np.random.normal(0, iv, n)))
    low = close * (1 - np.abs(np.random.normal(0, iv, n)))
    opn = np.roll(close, 1) * (1 + np.random.normal(0, dv * 0.3, n))
    opn[0] = initial_price
    df = pd.DataFrame({"open": opn, "high": high, "low": low, "close": close,
                        "volume": np.random.randint(5e7, 1.5e8, n)}, index=days)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def generate_iv_data(price_data: pd.DataFrame, base_iv: float = 0.20,
                     mean_reversion: float = 0.05, vol_of_vol: float = 0.03,
                     seed: int = 123) -> pd.DataFrame:
    np.random.seed(seed)
    n = len(price_data)
    rets = price_data["close"].pct_change().fillna(0)
    iv = np.zeros(n)
    iv[0] = base_iv
    for i in range(1, n):
        iv[i] = max(0.05, iv[i-1] + mean_reversion * (base_iv - iv[i-1])
                     - 0.5 * rets.iloc[i] + np.random.normal(0, vol_of_vol))
    return pd.DataFrame({"iv": iv}, index=price_data.index)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def trend_score(prices: pd.Series) -> float:
    """Returns -1 to +1. Positive = uptrend, negative = downtrend."""
    if len(prices) < 55:
        return 0.0
    current = prices.iloc[-1]
    s20 = sma(prices, 20).iloc[-1]
    s50 = sma(prices, 50).iloc[-1]
    e10 = ema(prices, 10).iloc[-1]

    # Score components
    score = 0.0
    score += 0.3 * np.sign(current - s50)     # Above/below 50 SMA
    score += 0.3 * np.sign(current - s20)     # Above/below 20 SMA
    score += 0.2 * np.sign(e10 - s20)         # Fast/slow MA alignment
    score += 0.2 * np.sign(s20 - s50)         # Medium/slow MA alignment
    return float(np.clip(score, -1, 1))


def realized_vol(prices: pd.Series, window: int = 20) -> float:
    if len(prices) < window + 1:
        return 0.20
    return float(prices.pct_change().iloc[-window:].std() * np.sqrt(252))


# ---------------------------------------------------------------------------
# Strategy 1: Adaptive Credit Spreads (the star performer)
# ---------------------------------------------------------------------------

class AdaptiveCreditSpread(Strategy):
    """Sells OTM credit spreads aligned with the trend.

    Key design:
    - Uptrend → sell put spreads (high probability, trend acts as tailwind)
    - Downtrend → sell call spreads (collect premium as market falls)
    - Neutral → sell put spreads further OTM (defensive)
    - NEVER sell against the trend (no put spreads in downtrends!)
    - Adaptive strike distance: further OTM when vol is high
    - Take profit at 50% of credit, stop at 2x credit
    """

    def __init__(self, pricing: PricingEngine, wing_width: float = 10.0,
                 target_dte: int = 30, base_risk_pct: float = 0.025,
                 min_iv_rank: float = 60.0, max_open: int = 3,
                 cooldown: int = 12) -> None:
        super().__init__(name="adaptive_credit")
        self.pricing = pricing
        self.wing_width = wing_width
        self.target_dte = target_dte
        self.base_risk_pct = base_risk_pct
        self.min_iv_rank = min_iv_rank
        self.max_open = max_open
        self.cooldown = cooldown
        self._prices: list[float] = []
        self._ivs: list[float] = []
        self._days_since: int = 999

    async def evaluate(self, portfolio: PortfolioSnapshot) -> Signal:
        self._days_since += 1
        if self._days_since < self.cooldown:
            return Signal(signal_type=SignalType.HOLD)
        if len(portfolio.positions) // 2 >= self.max_open:
            return Signal(signal_type=SignalType.HOLD)
        if len(self._prices) < 55 or len(self._ivs) < 55:
            return Signal(signal_type=SignalType.HOLD)

        prices = pd.Series(self._prices)
        iv_series = pd.Series(self._ivs)
        spot = self._prices[-1]
        cur_iv = self._ivs[-1]

        # IV rank check
        iv_lo, iv_hi = iv_series.min(), iv_series.max()
        if iv_hi == iv_lo:
            return Signal(signal_type=SignalType.HOLD)
        ivr = (cur_iv - iv_lo) / (iv_hi - iv_lo) * 100
        if ivr < self.min_iv_rank:
            return Signal(signal_type=SignalType.HOLD, reason=f"IVR {ivr:.0f}")

        ts = trend_score(prices)
        rvol = realized_vol(prices)
        exp = date.today() + timedelta(days=self.target_dte)

        # Adaptive OTM distance: further out when vol is high
        vol_mult = max(cur_iv, rvol)
        std = spot * vol_mult * np.sqrt(self.target_dte / 365)
        otm_factor = 1.2 + max(0, vol_mult - 0.18) * 3  # 1.2 at 18% vol, 1.8 at 38%

        # Require a defined trend — no neutral sales. In high vol, demand stronger confluence.
        min_trend = 0.4 if rvol < 0.22 else 0.8
        if abs(ts) < min_trend:
            return Signal(signal_type=SignalType.HOLD, reason=f"Weak trend ({ts:.1f})")

        # Skip if realized vol is extreme — strikes will get blown through
        if rvol > 0.28:
            return Signal(signal_type=SignalType.HOLD, reason=f"Vol too high ({rvol:.0%})")

        if ts > 0:
            # Uptrend: sell put spread
            short_k = round((spot - std * otm_factor) / 5) * 5
            long_k = short_k - self.wing_width
            short_c = OptionContract("SPY", exp, short_k, OptionType.PUT)
            long_c = OptionContract("SPY", exp, long_k, OptionType.PUT)
            desc = f"bull put (trend={ts:.1f})"
        else:
            # Downtrend: sell call spread
            short_k = round((spot + std * otm_factor) / 5) * 5
            long_k = short_k + self.wing_width
            short_c = OptionContract("SPY", exp, short_k, OptionType.CALL)
            long_c = OptionContract("SPY", exp, long_k, OptionType.CALL)
            desc = f"bear call (trend={ts:.1f})"

        sp = self.pricing.price(short_c, spot, cur_iv)
        lp = self.pricing.price(long_c, spot, cur_iv)
        credit = sp.theoretical_price - lp.theoretical_price

        if credit < 0.25:
            return Signal(signal_type=SignalType.HOLD, reason="Low credit")

        max_loss = (self.wing_width - credit) * 100
        vol_scale = max(0.5, min(1.0, 0.20 / rvol))
        qty = max(1, min(int(portfolio.net_liquidation * self.base_risk_pct * vol_scale / max_loss), 10))

        orders = [
            Order(contract=long_c, side=Side.BUY, quantity=qty,
                  order_type=OrderType.LIMIT, limit_price=lp.theoretical_price),
            Order(contract=short_c, side=Side.SELL, quantity=qty,
                  order_type=OrderType.LIMIT, limit_price=sp.theoretical_price),
        ]

        self._days_since = 0
        sig = Signal(signal_type=SignalType.OPEN, orders=orders,
                     reason=f"{desc} IVR:{ivr:.0f}", confidence=0.5)
        sig._take_profit_pct = 0.50
        sig._stop_loss_pct = 1.5  # Tighter stop — cut losers before they compound
        return sig

    async def on_fill(self, order: Order) -> None:
        pass

    def feed_price(self, price: float, iv: float = 0.20) -> None:
        self._prices.append(price)
        self._ivs.append(iv)


# ---------------------------------------------------------------------------
# Strategy 2: Trend Debit Spreads (directional, only clear trends)
# ---------------------------------------------------------------------------

class TrendDebitSpread(Strategy):
    """Buys debit spreads in confirmed trends with strict filters.

    Only trades when:
    - Trend score is strong (>0.6 or <-0.6) = all 4 MA components aligned
    - RSI confirms (not overbought/oversold)
    - Realized vol is not extreme (avoids whipsaw regimes)
    """

    def __init__(self, pricing: PricingEngine, spread_width: float = 10.0,
                 target_dte: int = 35, base_risk_pct: float = 0.025,
                 max_open: int = 3, cooldown: int = 10) -> None:
        super().__init__(name="trend_debit")
        self.pricing = pricing
        self.spread_width = spread_width
        self.target_dte = target_dte
        self.base_risk_pct = base_risk_pct
        self.max_open = max_open
        self.cooldown = cooldown
        self._prices: list[float] = []
        self._ivs: list[float] = []
        self._days_since: int = 999

    async def evaluate(self, portfolio: PortfolioSnapshot) -> Signal:
        self._days_since += 1
        if self._days_since < self.cooldown:
            return Signal(signal_type=SignalType.HOLD)
        if len(portfolio.positions) // 2 >= self.max_open:
            return Signal(signal_type=SignalType.HOLD)
        if len(self._prices) < 55:
            return Signal(signal_type=SignalType.HOLD)

        prices = pd.Series(self._prices)
        spot = self._prices[-1]
        iv = self._ivs[-1] if self._ivs else 0.20

        ts = trend_score(prices)
        rvol = realized_vol(prices)

        # Only trade strong, confirmed trends (0.8 = full MA confluence)
        if abs(ts) < 0.8:
            return Signal(signal_type=SignalType.HOLD, reason=f"Weak trend ({ts:.1f})")

        # Skip extreme vol regimes (whipsaws destroy debit spreads)
        if rvol > 0.30:
            return Signal(signal_type=SignalType.HOLD, reason=f"Vol too high ({rvol:.0%})")

        # RSI filter — plus momentum confluence (20-day return must agree with trend)
        current_rsi = rsi(prices, 14).iloc[-1]
        if np.isnan(current_rsi):
            return Signal(signal_type=SignalType.HOLD)

        ret_20 = (prices.iloc[-1] / prices.iloc[-20] - 1) if len(prices) >= 20 else 0
        if ts > 0 and ret_20 <= 0:
            return Signal(signal_type=SignalType.HOLD, reason="Momentum disagrees")
        if ts < 0 and ret_20 >= 0:
            return Signal(signal_type=SignalType.HOLD, reason="Momentum disagrees")

        exp = date.today() + timedelta(days=self.target_dte)

        if ts > 0:
            if current_rsi > 72:
                return Signal(signal_type=SignalType.HOLD, reason="Overbought")
            long_k = round(spot / 5) * 5
            short_k = long_k + self.spread_width
            long_c = OptionContract("SPY", exp, long_k, OptionType.CALL)
            short_c = OptionContract("SPY", exp, short_k, OptionType.CALL)
            direction = "bull"
        else:
            if current_rsi < 28:
                return Signal(signal_type=SignalType.HOLD, reason="Oversold")
            long_k = round(spot / 5) * 5
            short_k = long_k - self.spread_width
            long_c = OptionContract("SPY", exp, long_k, OptionType.PUT)
            short_c = OptionContract("SPY", exp, short_k, OptionType.PUT)
            direction = "bear"

        lp = self.pricing.price(long_c, spot, iv)
        sp = self.pricing.price(short_c, spot, iv)
        cost = lp.theoretical_price - sp.theoretical_price

        if cost <= 0.15:
            return Signal(signal_type=SignalType.HOLD)

        vol_scale = max(0.5, min(1.0, 0.22 / rvol))
        risk = portfolio.net_liquidation * self.base_risk_pct * vol_scale
        qty = max(1, min(int(risk / (cost * 100)), 10))

        orders = [
            Order(contract=long_c, side=Side.BUY, quantity=qty,
                  order_type=OrderType.LIMIT, limit_price=lp.theoretical_price),
            Order(contract=short_c, side=Side.SELL, quantity=qty,
                  order_type=OrderType.LIMIT, limit_price=sp.theoretical_price),
        ]

        self._days_since = 0
        sig = Signal(signal_type=SignalType.OPEN, orders=orders,
                     reason=f"{'Bull' if direction == 'bull' else 'Bear'} debit (trend={ts:.1f}, RSI={current_rsi:.0f})",
                     confidence=abs(ts))
        sig._take_profit_pct = 0.60  # Let winners run (stronger trends = bigger moves)
        sig._stop_loss_pct = 1.2     # Tight stop — if trend breaks, exit fast
        return sig

    async def on_fill(self, order: Order) -> None:
        pass

    def feed_price(self, price: float, iv: float = 0.20) -> None:
        self._prices.append(price)
        self._ivs.append(iv)


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

class CombinedStrategy(Strategy):
    def __init__(self, *strategies: Strategy) -> None:
        super().__init__(name="combined_v3")
        self.strategies = list(strategies)

    async def evaluate(self, portfolio: PortfolioSnapshot) -> Signal:
        for s in self.strategies:
            sig = await s.evaluate(portfolio)
            if sig.signal_type != SignalType.HOLD:
                return sig
        return Signal(signal_type=SignalType.HOLD)

    async def on_fill(self, order: Order) -> None:
        pass

    def feed_price(self, price: float, iv: float = 0.20) -> None:
        for s in self.strategies:
            if hasattr(s, "feed_price"):
                s.feed_price(price, iv)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class DataFeedingBacktestEngine(BacktestEngine):
    def run(self, strategy, price_data, iv_data=None):
        dates = self._get_trading_dates(price_data)
        for d in dates:
            spot = float(price_data.loc[d]["close"])
            iv = float(iv_data.loc[d]["iv"]) if iv_data is not None and d in iv_data.index else 0.20
            if hasattr(strategy, "feed_price"):
                strategy.feed_price(spot, iv)
            self._mark_positions(d, spot, iv)
            self._check_exits(d, spot)
            portfolio = self._build_portfolio(d)
            try:
                loop = asyncio.new_event_loop()
                sig = loop.run_until_complete(strategy.evaluate(portfolio))
                loop.close()
            except Exception:
                sig = Signal(signal_type=SignalType.HOLD)
            if sig.signal_type != SignalType.HOLD:
                self._process_signal(sig, d, spot, iv)
            eq = self._calculate_equity()
            self._peak_equity = max(self._peak_equity, eq)
            dd = (self._peak_equity - eq) / self._peak_equity if self._peak_equity > 0 else 0
            self._equity_history.append({"date": d, "equity": eq, "drawdown": dd,
                                          "cash": self._cash, "positions_value": eq - self._cash})
        if dates:
            self._close_all_positions(dates[-1], float(price_data.loc[dates[-1]]["close"]))
        return self._compile_results()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backtest():
    print("=" * 70)
    print("  LEGENDARY CHAINSAW — Strategy Optimization v3")
    print("  Target: Positive Sharpe + High Win Rate in ALL regimes")
    print("=" * 70)
    print()

    start, end = date(2024, 1, 2), date(2025, 12, 31)
    print(f"  Period: {start} → {end}  |  Capital: $100,000\n")

    scenarios = {
        "Bull (+15%, 16% vol)": {"annual_return": 0.15, "annual_vol": 0.16, "base_iv": 0.18, "seed": 42},
        "Choppy (+3%, 22% vol)": {"annual_return": 0.03, "annual_vol": 0.22, "base_iv": 0.24, "seed": 99},
        "Bear (-12%, 28% vol)": {"annual_return": -0.12, "annual_vol": 0.28, "base_iv": 0.30, "seed": 77},
    }

    pricing = PricingEngine(risk_free_rate=0.05)
    all_results = {}

    for sname, p in scenarios.items():
        print(f"--- {sname} ---")
        pd_data = generate_stock_data("SPY", start, end, 450.0, p["annual_return"], p["annual_vol"], p["seed"])
        iv_data = generate_iv_data(pd_data, p["base_iv"], seed=p["seed"] + 1)

        strats = {
            "Adaptive Credit": lambda: AdaptiveCreditSpread(pricing),
            "Trend Debit": lambda: TrendDebitSpread(pricing),
            "Combined": lambda: CombinedStrategy(
                AdaptiveCreditSpread(pricing, base_risk_pct=0.018),
                TrendDebitSpread(pricing, base_risk_pct=0.018),
            ),
        }

        scenario_results = {}
        for sn, sf in strats.items():
            cfg = BacktestConfig(start_date=start, end_date=end, initial_capital=100_000)
            eng = DataFeedingBacktestEngine(cfg, pricing)
            r = eng.run(sf(), pd_data, iv_data)
            scenario_results[sn] = r
            tag = " PASS" if r.sharpe_ratio > 0 else " FAIL"
            exits = {}
            for t in r.trades:
                k = "TP" if t.exit_reason.startswith("take_profit") else t.exit_reason[:6]
                exits[k] = exits.get(k, 0) + 1
            ex_str = " ".join(f"{k}:{v}" for k, v in sorted(exits.items()))
            print(f"  {sn:<18}{tag}  Ret:{r.total_return:>7.2%}  Sharpe:{r.sharpe_ratio:>5.2f}  "
                  f"Win:{r.win_rate:>4.0%}  DD:{r.max_drawdown:>6.2%}  PF:{r.profit_factor:>5.2f}  "
                  f"N:{r.total_trades:>3}  [{ex_str}]")

        all_results[sname] = scenario_results
        print()

    # Summary
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    skeys = list(scenarios.keys())
    short_names = ["Bull", "Choppy", "Bear"]
    print(f"\n  {'Strategy':<18} {'Bull':>8} {'Choppy':>8} {'Bear':>8} {'Pass?':>6}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for sn in ["Adaptive Credit", "Trend Debit", "Combined"]:
        vals = [all_results[sk][sn].sharpe_ratio for sk in skeys]
        ok = "YES" if all(v > 0 for v in vals) else "no"
        print(f"  {sn:<18} {vals[0]:>7.2f} {vals[1]:>7.2f} {vals[2]:>7.2f} {ok:>6}")

    print("\n  RECOMMENDED (positive Sharpe everywhere):")
    any_rec = False
    for sn in ["Adaptive Credit", "Trend Debit", "Combined"]:
        vals = [all_results[sk][sn] for sk in skeys]
        if all(v.sharpe_ratio > 0 for v in vals):
            any_rec = True
            avg_ret = np.mean([v.total_return for v in vals])
            avg_sharpe = np.mean([v.sharpe_ratio for v in vals])
            avg_win = np.mean([v.win_rate for v in vals])
            avg_dd = np.max([v.max_drawdown for v in vals])
            print(f"  >> {sn}: Avg Return {avg_ret:.2%} | Avg Sharpe {avg_sharpe:.2f} | "
                  f"Avg Win {avg_win:.0%} | Worst DD {avg_dd:.2%}")
            for i, sk in enumerate(skeys):
                v = vals[i]
                print(f"     {short_names[i]:<8} Ret:{v.total_return:>7.2%}  Sharpe:{v.sharpe_ratio:>5.2f}  "
                      f"Win:{v.win_rate:>4.0%}  DD:{v.max_drawdown:>6.2%}  PF:{v.profit_factor:>5.2f}  N:{v.total_trades}")

    if not any_rec:
        print("  None passed all scenarios. Closest candidates above.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_backtest()
