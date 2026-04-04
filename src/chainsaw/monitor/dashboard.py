"""Real-time portfolio monitoring and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from chainsaw.logging import get_logger
from chainsaw.models import OptionContract, PortfolioSnapshot

log = get_logger(__name__)


@dataclass
class DailyPnL:
    date: str
    starting_equity: float
    ending_equity: float
    pnl: float
    pnl_pct: float
    max_drawdown: float


@dataclass
class PositionReport:
    contract: str
    quantity: int
    avg_cost: float
    market_price: float
    unrealized_pnl: float
    delta: float
    gamma: float
    theta: float
    vega: float


class PortfolioDashboard:
    """Tracks portfolio state over time and generates reports."""

    def __init__(self) -> None:
        self._snapshots: list[PortfolioSnapshot] = []
        self._daily_pnl: list[DailyPnL] = []
        self._daily_start_equity: float | None = None
        self._peak_equity: float = 0.0
        self._alerts: list[dict] = []

    def record_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Record a portfolio snapshot. Call on every tick."""
        self._snapshots.append(snapshot)

        if self._daily_start_equity is None:
            self._daily_start_equity = snapshot.net_liquidation

        self._peak_equity = max(self._peak_equity, snapshot.net_liquidation)

        # Check for alert conditions
        self._check_alerts(snapshot)

    def end_of_day(self, snapshot: PortfolioSnapshot) -> DailyPnL:
        """Record end-of-day P&L. Call at market close."""
        start = self._daily_start_equity or snapshot.net_liquidation
        pnl = snapshot.net_liquidation - start
        pnl_pct = pnl / start if start > 0 else 0

        daily_max_dd = 0.0
        for s in self._snapshots:
            dd = (start - s.net_liquidation) / start if start > 0 else 0
            daily_max_dd = max(daily_max_dd, dd)

        record = DailyPnL(
            date=snapshot.timestamp.strftime("%Y-%m-%d"),
            starting_equity=start,
            ending_equity=snapshot.net_liquidation,
            pnl=pnl,
            pnl_pct=pnl_pct,
            max_drawdown=daily_max_dd,
        )
        self._daily_pnl.append(record)
        self._daily_start_equity = None
        self._snapshots.clear()

        return record

    def get_positions_report(self, snapshot: PortfolioSnapshot) -> list[PositionReport]:
        """Generate a detailed positions report."""
        reports = []
        for pos in snapshot.positions:
            reports.append(PositionReport(
                contract=str(pos.contract),
                quantity=pos.quantity,
                avg_cost=pos.avg_cost,
                market_price=pos.market_price,
                unrealized_pnl=pos.unrealized_pnl,
                delta=pos.greeks.delta,
                gamma=pos.greeks.gamma,
                theta=pos.greeks.theta,
                vega=pos.greeks.vega,
            ))
        return reports

    def get_greeks_summary(self, snapshot: PortfolioSnapshot) -> dict:
        """Portfolio-level Greeks summary."""
        return {
            "net_delta": snapshot.total_delta,
            "net_gamma": snapshot.total_gamma,
            "net_theta": snapshot.total_theta,
            "net_vega": snapshot.total_vega,
            "position_count": len(snapshot.positions),
        }

    def get_risk_summary(self, snapshot: PortfolioSnapshot) -> dict:
        """Portfolio risk metrics."""
        drawdown_from_peak = (
            (self._peak_equity - snapshot.net_liquidation) / self._peak_equity
            if self._peak_equity > 0 else 0
        )

        return {
            "net_liquidation": snapshot.net_liquidation,
            "cash": snapshot.cash,
            "margin_used": snapshot.margin_used,
            "margin_available": snapshot.margin_available,
            "margin_utilization": f"{snapshot.margin_utilization:.1%}",
            "peak_equity": self._peak_equity,
            "drawdown_from_peak": f"{drawdown_from_peak:.2%}",
        }

    def format_status(self, snapshot: PortfolioSnapshot) -> str:
        """Generate a formatted text status report."""
        risk = self.get_risk_summary(snapshot)
        greeks = self.get_greeks_summary(snapshot)
        positions = self.get_positions_report(snapshot)

        lines = [
            "=" * 60,
            f"  PORTFOLIO STATUS — {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "  ACCOUNT",
            f"    Net Liquidation:  ${risk['net_liquidation']:>12,.2f}",
            f"    Cash:             ${risk['cash']:>12,.2f}",
            f"    Margin Used:      ${risk['margin_used']:>12,.2f}",
            f"    Margin Available: ${risk['margin_available']:>12,.2f}",
            f"    Margin Util:       {risk['margin_utilization']:>12}",
            f"    Peak Equity:      ${risk['peak_equity']:>12,.2f}",
            f"    Drawdown:          {risk['drawdown_from_peak']:>12}",
            "",
            "  GREEKS",
            f"    Delta: {greeks['net_delta']:>10.2f}",
            f"    Gamma: {greeks['net_gamma']:>10.4f}",
            f"    Theta: {greeks['net_theta']:>10.2f}",
            f"    Vega:  {greeks['net_vega']:>10.2f}",
            "",
        ]

        if positions:
            lines.append("  POSITIONS")
            lines.append(f"    {'Contract':<30} {'Qty':>5} {'Price':>8} {'P&L':>10} {'Delta':>7}")
            lines.append("    " + "-" * 62)
            for p in positions:
                lines.append(
                    f"    {p.contract:<30} {p.quantity:>5} {p.market_price:>8.2f} "
                    f"${p.unrealized_pnl:>9.2f} {p.delta:>7.3f}"
                )
        else:
            lines.append("  No open positions.")

        if self._alerts:
            lines.append("")
            lines.append("  ALERTS")
            for alert in self._alerts[-5:]:
                lines.append(f"    [{alert['level']}] {alert['message']}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def _check_alerts(self, snapshot: PortfolioSnapshot) -> None:
        """Generate alerts for concerning conditions."""
        if snapshot.margin_utilization > 0.60:
            self._add_alert("WARNING", f"Margin utilization at {snapshot.margin_utilization:.1%}")

        if abs(snapshot.total_delta) > 300:
            self._add_alert("WARNING", f"Portfolio delta elevated: {snapshot.total_delta:.1f}")

        if self._peak_equity > 0:
            dd = (self._peak_equity - snapshot.net_liquidation) / self._peak_equity
            if dd > 0.02:
                self._add_alert("CRITICAL", f"Drawdown {dd:.2%} from peak")

    def _add_alert(self, level: str, message: str) -> None:
        self._alerts.append({
            "level": level,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })
        if level == "CRITICAL":
            log.critical("portfolio_alert", message=message)
        else:
            log.warning("portfolio_alert", message=message)
