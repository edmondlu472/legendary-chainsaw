"""Tests for the pricing engine."""

from datetime import date, timedelta

from chainsaw.models import OptionContract, OptionType
from chainsaw.pricing.engine import PricingEngine


def _make_contract(strike: float, option_type: OptionType, dte: int = 30) -> OptionContract:
    return OptionContract(
        symbol="SPY",
        expiration=date.today() + timedelta(days=dte),
        strike=strike,
        option_type=option_type,
    )


def test_call_price_positive():
    engine = PricingEngine(risk_free_rate=0.05)
    contract = _make_contract(450.0, OptionType.CALL, dte=30)
    result = engine.price(contract, spot=455.0, iv=0.20)
    assert result.theoretical_price > 0


def test_put_price_positive():
    engine = PricingEngine(risk_free_rate=0.05)
    contract = _make_contract(450.0, OptionType.PUT, dte=30)
    result = engine.price(contract, spot=445.0, iv=0.20)
    assert result.theoretical_price > 0


def test_call_delta_positive():
    engine = PricingEngine()
    contract = _make_contract(450.0, OptionType.CALL, dte=30)
    result = engine.price(contract, spot=450.0, iv=0.20)
    assert 0 < result.greeks.delta < 1


def test_put_delta_negative():
    engine = PricingEngine()
    contract = _make_contract(450.0, OptionType.PUT, dte=30)
    result = engine.price(contract, spot=450.0, iv=0.20)
    assert -1 < result.greeks.delta < 0


def test_gamma_positive():
    engine = PricingEngine()
    contract = _make_contract(450.0, OptionType.CALL, dte=30)
    result = engine.price(contract, spot=450.0, iv=0.20)
    assert result.greeks.gamma > 0


def test_theta_negative_for_long():
    engine = PricingEngine()
    contract = _make_contract(450.0, OptionType.CALL, dte=30)
    result = engine.price(contract, spot=450.0, iv=0.20)
    assert result.greeks.theta < 0


def test_vega_positive():
    engine = PricingEngine()
    contract = _make_contract(450.0, OptionType.CALL, dte=30)
    result = engine.price(contract, spot=450.0, iv=0.20)
    assert result.greeks.vega > 0


def test_implied_vol_roundtrip():
    engine = PricingEngine()
    contract = _make_contract(450.0, OptionType.CALL, dte=30)
    original_iv = 0.25
    result = engine.price(contract, spot=450.0, iv=original_iv)
    computed_iv = engine.implied_vol(contract, spot=450.0, market_price=result.theoretical_price)
    assert abs(computed_iv - original_iv) < 0.01


def test_expired_option_intrinsic():
    engine = PricingEngine()
    contract = _make_contract(450.0, OptionType.CALL, dte=0)
    result = engine.price(contract, spot=455.0, iv=0.20)
    assert result.theoretical_price == 5.0


def test_spread_price():
    engine = PricingEngine()
    long = _make_contract(450.0, OptionType.CALL, dte=30)
    short = _make_contract(455.0, OptionType.CALL, dte=30)
    spread = engine.spread_price(long, short, spot=452.0, long_iv=0.20, short_iv=0.20)
    assert spread > 0
