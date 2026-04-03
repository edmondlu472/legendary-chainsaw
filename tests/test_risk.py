"""Tests for the risk management module."""

from datetime import datetime

from chainsaw.config import RiskConfig
from chainsaw.models import (
    Greeks,
    Order,
    OrderType,
    PortfolioSnapshot,
    Position,
    Side,
)
from chainsaw.risk.manager import RiskManager


def _make_portfolio(
    nlv: float = 100_000,
    margin_used: float = 20_000,
    margin_avail: float = 80_000,
    delta: float = 0,
    vega: float = 0,
) -> PortfolioSnapshot:
    positions = []
    if delta or vega:
        positions.append(Position(
            contract="SPY",
            quantity=1,
            avg_cost=0,
            greeks=Greeks(delta=delta, vega=vega),
        ))
    return PortfolioSnapshot(
        timestamp=datetime.now(),
        net_liquidation=nlv,
        cash=nlv - margin_used,
        margin_used=margin_used,
        margin_available=margin_avail,
        positions=positions,
    )


def _make_order(limit_price: float = 2.0, qty: int = 1) -> Order:
    return Order(
        contract="SPY",
        side=Side.BUY,
        quantity=qty,
        order_type=OrderType.LIMIT,
        limit_price=limit_price,
    )


def test_order_approved_within_limits():
    rm = RiskManager()
    rm.reset_daily(100_000)
    portfolio = _make_portfolio()
    result = rm.check_order(_make_order(), portfolio)
    assert result.approved


def test_order_rejected_margin_exceeded():
    config = RiskConfig(max_margin_utilization=0.50)
    rm = RiskManager(config)
    rm.reset_daily(100_000)
    portfolio = _make_portfolio(margin_used=60_000, margin_avail=40_000)
    result = rm.check_order(_make_order(), portfolio)
    assert not result.approved
    assert "Margin" in result.reason


def test_kill_switch_on_drawdown():
    config = RiskConfig(max_daily_drawdown_pct=0.02)
    rm = RiskManager(config)
    rm.reset_daily(100_000)
    portfolio = _make_portfolio(nlv=97_000)
    result = rm.monitor_portfolio(portfolio)
    assert not result.approved
    assert rm.is_halted


def test_halted_blocks_orders():
    rm = RiskManager()
    rm._halted = True
    portfolio = _make_portfolio()
    result = rm.check_order(_make_order(), portfolio)
    assert not result.approved
    assert "kill switch" in result.reason.lower()


def test_delta_limit_warning():
    config = RiskConfig(max_portfolio_delta=100)
    rm = RiskManager(config)
    rm.reset_daily(100_000)
    portfolio = _make_portfolio(delta=150)
    result = rm.check_order(_make_order(), portfolio)
    assert not result.approved
    assert "delta" in result.reason.lower()


def test_vega_limit_warning():
    config = RiskConfig(max_portfolio_vega=5000)
    rm = RiskManager(config)
    rm.reset_daily(100_000)
    portfolio = _make_portfolio(vega=6000)
    result = rm.check_order(_make_order(), portfolio)
    assert not result.approved
    assert "vega" in result.reason.lower()
