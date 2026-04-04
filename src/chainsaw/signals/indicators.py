"""Technical indicators and signal generators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower) Bollinger Bands."""
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


# --- Signal Functions ---
# Each returns a float between -1.0 (strong bearish) and 1.0 (strong bullish)


def momentum_signal(
    prices: pd.Series,
    fast_period: int = 10,
    slow_period: int = 30,
    rsi_period: int = 14,
) -> float:
    """Combined momentum signal from moving average crossover + RSI.

    Returns:
        Float between -1.0 and 1.0.
        > 0.3 = bullish, < -0.3 = bearish, between = neutral.
    """
    if len(prices) < slow_period + 5:
        return 0.0

    fast_ma = ema(prices, fast_period)
    slow_ma = ema(prices, slow_period)
    current_rsi = rsi(prices, rsi_period)

    # MA crossover component (-1 to 1)
    ma_diff = (fast_ma.iloc[-1] - slow_ma.iloc[-1]) / slow_ma.iloc[-1]
    ma_signal = np.clip(ma_diff * 20, -1.0, 1.0)  # Scale so 5% diff = max signal

    # RSI component (-1 to 1)
    rsi_val = current_rsi.iloc[-1]
    if np.isnan(rsi_val):
        rsi_signal = 0.0
    elif rsi_val > 70:
        rsi_signal = -((rsi_val - 70) / 30)  # Overbought = bearish
    elif rsi_val < 30:
        rsi_signal = (30 - rsi_val) / 30  # Oversold = bullish
    else:
        rsi_signal = (rsi_val - 50) / 50  # Neutral zone, slight lean

    # Weighted combination
    return float(np.clip(ma_signal * 0.6 + rsi_signal * 0.4, -1.0, 1.0))


def mean_reversion_signal(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> float:
    """Mean reversion signal using Bollinger Bands.

    Returns:
        Float between -1.0 and 1.0.
        Positive when price is below mean (expect reversion up).
        Negative when price is above mean (expect reversion down).
    """
    if len(prices) < period + 5:
        return 0.0

    upper, middle, lower = bollinger_bands(prices, period, num_std)
    current = prices.iloc[-1]
    mid = middle.iloc[-1]
    up = upper.iloc[-1]
    lo = lower.iloc[-1]

    if np.isnan(mid) or np.isnan(up) or np.isnan(lo):
        return 0.0

    band_width = up - lo
    if band_width <= 0:
        return 0.0

    # Position within bands: -1 at upper, +1 at lower
    position = (mid - current) / (band_width / 2)
    return float(np.clip(position, -1.0, 1.0))


def iv_rank(current_iv: float, iv_history: pd.Series) -> float:
    """IV Rank: where current IV sits relative to 52-week high/low.

    Returns 0-100 scale. High = IV is elevated = good for selling premium.
    """
    if iv_history.empty or len(iv_history) < 5:
        return 50.0

    high = iv_history.max()
    low = iv_history.min()

    if high == low:
        return 50.0

    return float(((current_iv - low) / (high - low)) * 100)


def iv_percentile(current_iv: float, iv_history: pd.Series) -> float:
    """IV Percentile: % of days in past year where IV was below current.

    Returns 0-100 scale. High = IV is elevated relative to history.
    """
    if iv_history.empty or len(iv_history) < 5:
        return 50.0

    below = (iv_history < current_iv).sum()
    return float((below / len(iv_history)) * 100)


def bollinger_band_signal(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> str:
    """Simple Bollinger Band breakout signal.

    Returns: "overbought", "oversold", or "neutral"
    """
    if len(prices) < period + 1:
        return "neutral"

    upper, middle, lower = bollinger_bands(prices, period, num_std)
    current = prices.iloc[-1]

    if current > upper.iloc[-1]:
        return "overbought"
    elif current < lower.iloc[-1]:
        return "oversold"
    return "neutral"
