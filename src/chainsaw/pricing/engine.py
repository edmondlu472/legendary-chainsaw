"""Options pricing engine using Black-Scholes (pure scipy implementation)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from scipy.stats import norm

from chainsaw.models import Greeks, OptionContract, OptionType


@dataclass
class PricingResult:
    theoretical_price: float
    greeks: Greeks
    iv: float


def _d1(s: float, k: float, t: float, r: float, sigma: float) -> float:
    return (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))


def _d2(s: float, k: float, t: float, r: float, sigma: float) -> float:
    return _d1(s, k, t, r, sigma) - sigma * math.sqrt(t)


def _bs_call_price(s: float, k: float, t: float, r: float, sigma: float) -> float:
    d1 = _d1(s, k, t, r, sigma)
    d2 = _d2(s, k, t, r, sigma)
    return s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)


def _bs_put_price(s: float, k: float, t: float, r: float, sigma: float) -> float:
    d1 = _d1(s, k, t, r, sigma)
    d2 = _d2(s, k, t, r, sigma)
    return k * math.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)


def _bs_price(flag: str, s: float, k: float, t: float, r: float, sigma: float) -> float:
    if flag == "c":
        return _bs_call_price(s, k, t, r, sigma)
    return _bs_put_price(s, k, t, r, sigma)


def _bs_delta(flag: str, s: float, k: float, t: float, r: float, sigma: float) -> float:
    d1 = _d1(s, k, t, r, sigma)
    if flag == "c":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


def _bs_gamma(s: float, k: float, t: float, r: float, sigma: float) -> float:
    d1 = _d1(s, k, t, r, sigma)
    return norm.pdf(d1) / (s * sigma * math.sqrt(t))


def _bs_theta(flag: str, s: float, k: float, t: float, r: float, sigma: float) -> float:
    d1 = _d1(s, k, t, r, sigma)
    d2 = _d2(s, k, t, r, sigma)
    term1 = -(s * norm.pdf(d1) * sigma) / (2 * math.sqrt(t))
    if flag == "c":
        return (term1 - r * k * math.exp(-r * t) * norm.cdf(d2)) / 365
    return (term1 + r * k * math.exp(-r * t) * norm.cdf(-d2)) / 365


def _bs_vega(s: float, k: float, t: float, r: float, sigma: float) -> float:
    d1 = _d1(s, k, t, r, sigma)
    return s * norm.pdf(d1) * math.sqrt(t) / 100  # Per 1% vol move


def _bs_rho(flag: str, s: float, k: float, t: float, r: float, sigma: float) -> float:
    d2 = _d2(s, k, t, r, sigma)
    if flag == "c":
        return k * t * math.exp(-r * t) * norm.cdf(d2) / 100
    return -k * t * math.exp(-r * t) * norm.cdf(-d2) / 100


class PricingEngine:
    """Compute option prices, Greeks, and implied volatility.

    Pure scipy implementation — no external options libraries required.
    """

    def __init__(self, risk_free_rate: float = 0.05) -> None:
        self.risk_free_rate = risk_free_rate

    def price(
        self,
        contract: OptionContract,
        spot: float,
        iv: float,
        rate: float | None = None,
    ) -> PricingResult:
        """Price an option and compute all Greeks."""
        r = rate or self.risk_free_rate
        t = self._time_to_expiry(contract.expiration)
        flag = "c" if contract.option_type == OptionType.CALL else "p"

        if t <= 0:
            intrinsic = self._intrinsic(contract, spot)
            return PricingResult(
                theoretical_price=intrinsic,
                greeks=Greeks(
                    delta=1.0 if intrinsic > 0 and flag == "c" else (-1.0 if intrinsic > 0 else 0.0),
                ),
                iv=0.0,
            )

        price = _bs_price(flag, spot, contract.strike, t, r, iv)
        greeks = Greeks(
            delta=_bs_delta(flag, spot, contract.strike, t, r, iv),
            gamma=_bs_gamma(spot, contract.strike, t, r, iv),
            theta=_bs_theta(flag, spot, contract.strike, t, r, iv),
            vega=_bs_vega(spot, contract.strike, t, r, iv),
            rho=_bs_rho(flag, spot, contract.strike, t, r, iv),
            iv=iv,
        )

        return PricingResult(theoretical_price=price, greeks=greeks, iv=iv)

    def implied_vol(
        self,
        contract: OptionContract,
        spot: float,
        market_price: float,
        rate: float | None = None,
    ) -> float:
        """Compute implied volatility from market price using bisection."""
        r = rate or self.risk_free_rate
        t = self._time_to_expiry(contract.expiration)
        flag = "c" if contract.option_type == OptionType.CALL else "p"

        if t <= 0 or market_price <= 0:
            return 0.0

        return self._iv_bisection(flag, spot, contract.strike, t, r, market_price)

    def spread_price(
        self,
        long_contract: OptionContract,
        short_contract: OptionContract,
        spot: float,
        long_iv: float,
        short_iv: float,
    ) -> float:
        """Price a vertical spread (long - short)."""
        long_result = self.price(long_contract, spot, long_iv)
        short_result = self.price(short_contract, spot, short_iv)
        return long_result.theoretical_price - short_result.theoretical_price

    def _time_to_expiry(self, expiration: date) -> float:
        """Time to expiry in years."""
        days = (expiration - date.today()).days
        return max(days / 365.0, 0.0)

    def _intrinsic(self, contract: OptionContract, spot: float) -> float:
        if contract.option_type == OptionType.CALL:
            return max(spot - contract.strike, 0.0)
        return max(contract.strike - spot, 0.0)

    def _iv_bisection(
        self,
        flag: str,
        spot: float,
        strike: float,
        t: float,
        r: float,
        target_price: float,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> float:
        """Bisection method for implied volatility."""
        low, high = 0.001, 5.0
        for _ in range(max_iter):
            mid = (low + high) / 2
            price = _bs_price(flag, spot, strike, t, r, mid)
            if abs(price - target_price) < tol:
                return mid
            if price > target_price:
                high = mid
            else:
                low = mid
        return (low + high) / 2
