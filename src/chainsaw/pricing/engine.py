"""Options pricing engine using Black-Scholes and py_vollib."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from scipy.stats import norm
from py_vollib.black_scholes import black_scholes as bs_price
from py_vollib.black_scholes.greeks.analytical import delta as bs_delta
from py_vollib.black_scholes.greeks.analytical import gamma as bs_gamma
from py_vollib.black_scholes.greeks.analytical import theta as bs_theta
from py_vollib.black_scholes.greeks.analytical import vega as bs_vega
from py_vollib.black_scholes.greeks.analytical import rho as bs_rho
from py_vollib.black_scholes.implied_volatility import implied_volatility as bs_iv

from chainsaw.models import Greeks, OptionContract, OptionType


@dataclass
class PricingResult:
    theoretical_price: float
    greeks: Greeks
    iv: float


class PricingEngine:
    """Compute option prices, Greeks, and implied volatility."""

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
            # Expired — return intrinsic value
            intrinsic = self._intrinsic(contract, spot)
            return PricingResult(
                theoretical_price=intrinsic,
                greeks=Greeks(
                    delta=1.0 if intrinsic > 0 and flag == "c" else (-1.0 if intrinsic > 0 else 0.0),
                ),
                iv=0.0,
            )

        price = bs_price(flag, spot, contract.strike, t, r, iv)
        greeks = Greeks(
            delta=bs_delta(flag, spot, contract.strike, t, r, iv),
            gamma=bs_gamma(flag, spot, contract.strike, t, r, iv),
            theta=bs_theta(flag, spot, contract.strike, t, r, iv),
            vega=bs_vega(flag, spot, contract.strike, t, r, iv),
            rho=bs_rho(flag, spot, contract.strike, t, r, iv),
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
        """Compute implied volatility from market price."""
        r = rate or self.risk_free_rate
        t = self._time_to_expiry(contract.expiration)
        flag = "c" if contract.option_type == OptionType.CALL else "p"

        if t <= 0 or market_price <= 0:
            return 0.0

        try:
            return bs_iv(market_price, spot, contract.strike, t, r, flag)
        except Exception:
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
        """Fallback bisection method for IV when Newton's method fails."""
        low, high = 0.001, 5.0
        for _ in range(max_iter):
            mid = (low + high) / 2
            price = bs_price(flag, spot, strike, t, r, mid)
            if abs(price - target_price) < tol:
                return mid
            if price > target_price:
                high = mid
            else:
                low = mid
        return (low + high) / 2
