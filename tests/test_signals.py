"""Tests for signal generation indicators and composite generator."""

import numpy as np
import pandas as pd

from chainsaw.signals.indicators import (
    bollinger_band_signal,
    ema,
    iv_percentile,
    iv_rank,
    mean_reversion_signal,
    momentum_signal,
    rsi,
    sma,
)
from chainsaw.signals.composite import CompositeSignalGenerator


def _trending_up(n: int = 100) -> pd.Series:
    """Generate an upward trending price series."""
    np.random.seed(42)
    returns = np.random.normal(0.002, 0.01, n)
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices)


def _trending_down(n: int = 100) -> pd.Series:
    np.random.seed(42)
    returns = np.random.normal(-0.002, 0.01, n)
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices)


def _flat(n: int = 100) -> pd.Series:
    np.random.seed(42)
    returns = np.random.normal(0.0, 0.005, n)
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices)


def test_sma():
    prices = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    result = sma(prices, 3)
    assert abs(result.iloc[-1] - 9.0) < 0.01


def test_ema():
    prices = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    result = ema(prices, 3)
    assert result.iloc[-1] > 8.0  # EMA should be close to recent prices


def test_rsi_bounds():
    prices = _trending_up()
    result = rsi(prices)
    valid = result.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_momentum_bullish_for_uptrend():
    prices = _trending_up()
    signal = momentum_signal(prices)
    assert signal > 0


def test_momentum_bearish_for_downtrend():
    prices = _trending_down()
    signal = momentum_signal(prices)
    assert signal < 0


def test_momentum_bounded():
    prices = _trending_up(200)
    signal = momentum_signal(prices)
    assert -1.0 <= signal <= 1.0


def test_mean_reversion_after_drop():
    # Price drops sharply — mean reversion should be bullish
    prices = pd.Series([100.0] * 50 + [90.0] * 5)
    signal = mean_reversion_signal(prices)
    assert signal > 0  # Expect reversion upward


def test_mean_reversion_bounded():
    prices = _trending_up()
    signal = mean_reversion_signal(prices)
    assert -1.0 <= signal <= 1.0


def test_iv_rank_high():
    history = pd.Series([0.15, 0.18, 0.20, 0.22, 0.16, 0.19, 0.17])
    rank = iv_rank(0.22, history)
    assert rank > 80


def test_iv_rank_low():
    history = pd.Series([0.15, 0.18, 0.20, 0.22, 0.16, 0.19, 0.17])
    rank = iv_rank(0.15, history)
    assert rank < 20


def test_iv_percentile_high():
    history = pd.Series([0.10, 0.12, 0.14, 0.16, 0.18, 0.20])
    pct = iv_percentile(0.19, history)
    assert pct > 60


def test_bollinger_overbought():
    prices = pd.Series([100.0] * 25 + [120.0])
    result = bollinger_band_signal(prices)
    assert result == "overbought"


def test_bollinger_oversold():
    prices = pd.Series([100.0] * 25 + [80.0])
    result = bollinger_band_signal(prices)
    assert result == "oversold"


def test_composite_bullish():
    gen = CompositeSignalGenerator()
    prices = _trending_up()
    output = gen.generate(prices)
    assert output.direction == "bull"
    assert output.strength > 0


def test_composite_bearish():
    gen = CompositeSignalGenerator()
    prices = _trending_down()
    output = gen.generate(prices)
    assert output.direction == "bear"
    assert output.strength > 0


def test_composite_should_sell_premium():
    gen = CompositeSignalGenerator(min_iv_rank_for_selling=30)
    history = pd.Series([0.15, 0.18, 0.20, 0.22, 0.16, 0.19, 0.17])
    assert gen.should_sell_premium(0.22, history)  # IV at top of range
    assert not gen.should_sell_premium(0.15, history)  # IV at bottom
