"""Risk management engine — enforces all position and portfolio limits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from chainsaw.config import RiskConfig
from chainsaw.logging import get_logger
from chainsaw.models import Order, PortfolioSnapshot, Side

log = get_logger(__name__)


@dataclass
class RiskCheckResult:
    approved: bool
    reason: str = ""


class RiskManager:
    """Validates orders and monitors portfolio risk in real-time."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self._config = config or RiskConfig()
        self._daily_start_equity: float | None = None
        self._daily_start_time: datetime | None = None
        self._halted = False

    @property
    def is_halted(self) -> bool:
        return self._halted

    def reset_daily(self, current_equity: float) -> None:
        """Call at start of each trading day to reset daily drawdown tracking."""
        self._daily_start_equity = current_equity
        self._daily_start_time = datetime.now()
        self._halted = False
        log.info("daily_risk_reset", equity=current_equity)

    def check_order(self, order: Order, portfolio: PortfolioSnapshot) -> RiskCheckResult:
        """Run all pre-trade risk checks on an order."""
        if self._halted:
            return RiskCheckResult(False, "Trading halted — kill switch active")

        # Check margin utilization
        if portfolio.margin_utilization > self._config.max_margin_utilization:
            return RiskCheckResult(
                False,
                f"Margin utilization {portfolio.margin_utilization:.1%} exceeds "
                f"limit {self._config.max_margin_utilization:.1%}",
            )

        # Check position size limit
        if portfolio.net_liquidation > 0:
            # Estimate order value (rough — would need pricing for exact)
            est_value = order.quantity * (order.limit_price or 0) * 100  # options multiplier
            position_pct = est_value / portfolio.net_liquidation
            if position_pct > self._config.max_position_pct:
                return RiskCheckResult(
                    False,
                    f"Position size {position_pct:.1%} exceeds limit {self._config.max_position_pct:.1%}",
                )

        # Check daily drawdown
        drawdown_check = self._check_daily_drawdown(portfolio)
        if not drawdown_check.approved:
            return drawdown_check

        # Check portfolio Greeks limits
        greeks_check = self._check_greeks(portfolio)
        if not greeks_check.approved:
            return greeks_check

        log.info("risk_check_passed", order=str(order.contract), side=order.side.value)
        return RiskCheckResult(True)

    def monitor_portfolio(self, portfolio: PortfolioSnapshot) -> RiskCheckResult:
        """Continuous portfolio monitoring. Call on every tick/update."""
        drawdown_check = self._check_daily_drawdown(portfolio)
        if not drawdown_check.approved:
            self._halt(drawdown_check.reason)
            return drawdown_check

        greeks_check = self._check_greeks(portfolio)
        if not greeks_check.approved:
            log.warning("greeks_limit_warning", reason=greeks_check.reason)

        return RiskCheckResult(True)

    def _check_daily_drawdown(self, portfolio: PortfolioSnapshot) -> RiskCheckResult:
        if self._daily_start_equity is None:
            return RiskCheckResult(True)

        drawdown = (self._daily_start_equity - portfolio.net_liquidation) / self._daily_start_equity
        if drawdown > self._config.max_daily_drawdown_pct:
            return RiskCheckResult(
                False,
                f"Daily drawdown {drawdown:.2%} exceeds limit {self._config.max_daily_drawdown_pct:.2%}",
            )
        return RiskCheckResult(True)

    def _check_greeks(self, portfolio: PortfolioSnapshot) -> RiskCheckResult:
        if abs(portfolio.total_delta) > self._config.max_portfolio_delta:
            return RiskCheckResult(
                False,
                f"Portfolio delta {portfolio.total_delta:.1f} exceeds limit "
                f"±{self._config.max_portfolio_delta:.1f}",
            )

        if abs(portfolio.total_vega) > self._config.max_portfolio_vega:
            return RiskCheckResult(
                False,
                f"Portfolio vega {portfolio.total_vega:.1f} exceeds limit "
                f"±{self._config.max_portfolio_vega:.1f}",
            )

        return RiskCheckResult(True)

    def _halt(self, reason: str) -> None:
        self._halted = True
        log.critical("kill_switch_activated", reason=reason)
