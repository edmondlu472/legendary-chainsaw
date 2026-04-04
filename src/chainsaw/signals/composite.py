"""Composite signal generator combining multiple indicators."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from chainsaw.signals.indicators import (
    iv_rank,
    iv_percentile,
    mean_reversion_signal,
    momentum_signal,
)


@dataclass
class SignalOutput:
    direction: str  # "bull", "bear", or "neutral"
    strength: float  # 0.0 to 1.0
    iv_rank: float  # 0-100
    iv_percentile: float  # 0-100
    momentum: float  # -1 to 1
    mean_reversion: float  # -1 to 1
    rationale: str = ""


class CompositeSignalGenerator:
    """Combines momentum, mean reversion, and IV signals into a single output.

    Signal logic:
    - For directional strategies (verticals): use momentum + mean reversion
    - For premium selling (condors): use IV rank/percentile as primary filter
    """

    def __init__(
        self,
        momentum_weight: float = 0.5,
        mean_reversion_weight: float = 0.3,
        iv_weight: float = 0.2,
        bull_threshold: float = 0.25,
        bear_threshold: float = -0.25,
        min_iv_rank_for_selling: float = 30.0,
    ) -> None:
        self.momentum_weight = momentum_weight
        self.mean_reversion_weight = mean_reversion_weight
        self.iv_weight = iv_weight
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.min_iv_rank_for_selling = min_iv_rank_for_selling

    def generate(
        self,
        prices: pd.Series,
        current_iv: float = 0.0,
        iv_history: pd.Series | None = None,
    ) -> SignalOutput:
        """Generate composite signal from price and IV data.

        Args:
            prices: Historical close prices (at least 50 bars recommended).
            current_iv: Current implied volatility of the underlying.
            iv_history: Historical IV values for rank/percentile calculation.
        """
        mom = momentum_signal(prices)
        mr = mean_reversion_signal(prices)

        if iv_history is not None and not iv_history.empty and current_iv > 0:
            ivr = iv_rank(current_iv, iv_history)
            ivp = iv_percentile(current_iv, iv_history)
            # High IV = favor selling = slight bearish lean (sell calls) or neutral (sell both sides)
            iv_signal = (ivr - 50) / 100  # -0.5 to 0.5, positive when IV is high
        else:
            ivr = 50.0
            ivp = 50.0
            iv_signal = 0.0

        # Weighted composite
        composite = (
            mom * self.momentum_weight +
            mr * self.mean_reversion_weight +
            iv_signal * self.iv_weight
        )

        # Determine direction
        if composite > self.bull_threshold:
            direction = "bull"
        elif composite < self.bear_threshold:
            direction = "bear"
        else:
            direction = "neutral"

        strength = min(abs(composite), 1.0)

        # Build rationale
        parts = []
        if abs(mom) > 0.2:
            parts.append(f"momentum {'bullish' if mom > 0 else 'bearish'} ({mom:.2f})")
        if abs(mr) > 0.2:
            parts.append(f"mean reversion {'bullish' if mr > 0 else 'bearish'} ({mr:.2f})")
        if ivr > 60:
            parts.append(f"IV elevated (rank {ivr:.0f})")
        elif ivr < 30:
            parts.append(f"IV low (rank {ivr:.0f})")
        rationale = "; ".join(parts) if parts else "no strong signals"

        return SignalOutput(
            direction=direction,
            strength=strength,
            iv_rank=ivr,
            iv_percentile=ivp,
            momentum=mom,
            mean_reversion=mr,
            rationale=rationale,
        )

    def should_sell_premium(self, current_iv: float, iv_history: pd.Series) -> bool:
        """Whether IV conditions favor premium selling strategies (condors, strangles)."""
        if iv_history.empty:
            return False
        return iv_rank(current_iv, iv_history) >= self.min_iv_rank_for_selling
