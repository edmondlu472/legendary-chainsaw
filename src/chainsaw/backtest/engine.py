"""Backtesting engine for options strategies with proper exit management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

from chainsaw.logging import get_logger
from chainsaw.models import (
    OptionContract,
    OptionType,
    Order,
    OrderType,
    PortfolioSnapshot,
    Position,
    Side,
)
from chainsaw.pricing.engine import PricingEngine
from chainsaw.strategy.base import Signal, SignalType, Strategy

log = get_logger(__name__)


@dataclass
class BacktestConfig:
    start_date: date
    end_date: date
    initial_capital: float = 100_000.0
    commission_per_contract: float = 0.65
    slippage_pct: float = 0.001  # 0.1% slippage on fills
    risk_free_rate: float = 0.05


@dataclass
class TradeRecord:
    entry_date: date
    exit_date: date | None = None
    strategy: str = ""
    description: str = ""
    quantity: int = 0
    entry_cost: float = 0.0
    exit_value: float = 0.0
    pnl: float = 0.0
    commission: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    config: BacktestConfig
    equity_curve: pd.DataFrame
    trades: list[TradeRecord]
    total_return: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_pnl: float = 0.0

    def summary(self) -> str:
        return (
            f"=== Backtest Results ===\n"
            f"Period:            {self.config.start_date} → {self.config.end_date}\n"
            f"Initial Capital:   ${self.config.initial_capital:,.0f}\n"
            f"Total Return:      {self.total_return:.2%}\n"
            f"Annualized Return: {self.annualized_return:.2%}\n"
            f"Max Drawdown:      {self.max_drawdown:.2%}\n"
            f"Sharpe Ratio:      {self.sharpe_ratio:.2f}\n"
            f"Win Rate:          {self.win_rate:.2%}\n"
            f"Profit Factor:     {self.profit_factor:.2f}\n"
            f"Total Trades:      {self.total_trades}\n"
            f"Avg Trade P&L:     ${self.avg_trade_pnl:,.2f}\n"
        )


@dataclass
class SpreadPosition:
    """Tracks a multi-leg spread as a single unit with exit management."""

    legs: list[LegPosition]
    entry_date: date
    entry_cost: float  # Net debit (positive) or credit (negative)
    max_profit: float
    max_loss: float
    take_profit_pct: float = 0.50  # Close at 50% of max profit
    stop_loss_pct: float = 2.0     # Close at 2x entry cost (for debits)
    description: str = ""
    quantity: int = 1

    @property
    def current_value(self) -> float:
        """Net current value of the spread (from holder's perspective)."""
        val = 0.0
        for leg in self.legs:
            if leg.side == Side.BUY:
                val += leg.current_price * leg.quantity * leg.contract.multiplier
            else:
                val -= leg.current_price * leg.quantity * leg.contract.multiplier
        return val

    @property
    def unrealized_pnl(self) -> float:
        return self.current_value - self.entry_cost

    @property
    def pnl_pct_of_max(self) -> float:
        """What % of max profit has been realized. 1.0 = full max profit."""
        if self.max_profit == 0:
            return 0.0
        return self.unrealized_pnl / self.max_profit

    @property
    def earliest_expiration(self) -> date:
        return min(leg.contract.expiration for leg in self.legs)

    def should_take_profit(self) -> bool:
        return self.pnl_pct_of_max >= self.take_profit_pct

    def should_stop_loss(self) -> bool:
        """Stop loss based on multiples of max_profit (credit received for credit spreads).

        For credit spread: stop_loss_pct=2.0 means close when loss = 2x credit.
        For debit spread: stop_loss_pct=0.8 means close when loss = 80% of cost.
        """
        if self.max_profit > 0 and self.unrealized_pnl < 0:
            loss_multiple = -self.unrealized_pnl / self.max_profit
            return loss_multiple >= self.stop_loss_pct
        return False

    def is_expired(self, current_date) -> bool:
        current = current_date.date() if hasattr(current_date, 'date') else current_date
        return self.earliest_expiration <= current

    def days_held(self, current_date) -> int:
        current = current_date.date() if hasattr(current_date, 'date') else current_date
        entry = self.entry_date.date() if hasattr(self.entry_date, 'date') else self.entry_date
        return (current - entry).days


@dataclass
class LegPosition:
    contract: OptionContract
    side: Side
    quantity: int
    entry_price: float
    current_price: float = 0.0


class BacktestEngine:
    """Event-driven backtester with spread-level exit management."""

    def __init__(self, config: BacktestConfig, pricing: PricingEngine | None = None) -> None:
        self.config = config
        self.pricing = pricing or PricingEngine(risk_free_rate=config.risk_free_rate)
        self._cash = config.initial_capital
        self._spreads: list[SpreadPosition] = []
        self._trades: list[TradeRecord] = []
        self._equity_history: list[dict] = []
        self._peak_equity = config.initial_capital

    def _get_trading_dates(self, price_data: pd.DataFrame) -> list:
        mask = (price_data.index >= pd.Timestamp(self.config.start_date)) & \
               (price_data.index <= pd.Timestamp(self.config.end_date))
        return list(price_data.index[mask])

    def _mark_positions(self, current_date, spot: float, iv: float) -> None:
        for spread in self._spreads:
            for leg in spread.legs:
                result = self.pricing.price(leg.contract, spot, iv)
                leg.current_price = result.theoretical_price

    def _check_exits(self, current_date, spot: float) -> None:
        """Check take-profit, stop-loss, and time-based exits for all spreads."""
        to_close = []
        for spread in self._spreads:
            reason = None

            if spread.is_expired(current_date):
                reason = "expiration"
            elif spread.should_take_profit():
                reason = f"take_profit ({spread.pnl_pct_of_max:.0%} of max)"
            elif spread.should_stop_loss():
                reason = "stop_loss"
            elif spread.days_held(current_date) >= 17 and spread.unrealized_pnl > 0:
                # Close profitable trades with <5 DTE to avoid gamma risk
                dte = (spread.earliest_expiration - (current_date.date() if hasattr(current_date, 'date') else current_date)).days
                if dte <= 5:
                    reason = "early_close_gamma_risk"

            if reason:
                to_close.append((spread, reason))

        for spread, reason in to_close:
            self._close_spread(spread, current_date, reason)

    def _close_spread(self, spread: SpreadPosition, current_date, reason: str) -> None:
        """Close a spread and record the trade."""
        exit_value = spread.current_value
        pnl = spread.unrealized_pnl
        commission = self.config.commission_per_contract * sum(l.quantity for l in spread.legs) * 2

        # Settle cash: reverse the position
        # If we paid entry_cost to open, we receive exit_value to close
        self._cash += exit_value - commission

        pnl -= commission

        d = current_date.date() if hasattr(current_date, 'date') else current_date
        ed = spread.entry_date.date() if hasattr(spread.entry_date, 'date') else spread.entry_date
        self._trades.append(TradeRecord(
            entry_date=ed,
            exit_date=d,
            description=spread.description,
            quantity=spread.quantity,
            entry_cost=spread.entry_cost,
            exit_value=exit_value,
            pnl=pnl,
            commission=commission,
            exit_reason=reason,
        ))
        self._spreads.remove(spread)

    def _process_signal(self, signal: Signal, current_date, spot: float, iv: float) -> None:
        """Process a signal by opening a new spread position."""
        orders = signal.orders
        if not orders or not all(isinstance(o.contract, OptionContract) for o in orders):
            return

        legs = []
        net_cost = 0.0

        for order in orders:
            result = self.pricing.price(order.contract, spot, iv)
            theo = result.theoretical_price

            if order.side == Side.BUY:
                fill_price = theo * (1 + self.config.slippage_pct)
                cost = fill_price * order.quantity * order.contract.multiplier
                commission = self.config.commission_per_contract * order.quantity
                self._cash -= cost + commission
                net_cost += cost
            else:
                fill_price = theo * (1 - self.config.slippage_pct)
                credit = fill_price * order.quantity * order.contract.multiplier
                commission = self.config.commission_per_contract * order.quantity
                self._cash += credit - commission
                net_cost -= credit

            legs.append(LegPosition(
                contract=order.contract,
                side=order.side,
                quantity=order.quantity,
                entry_price=fill_price,
                current_price=fill_price,
            ))

        # Determine max profit/loss based on spread type
        max_profit, max_loss = self._calc_spread_bounds(legs, net_cost)

        # Extract take-profit and stop-loss from signal metadata if present
        tp = getattr(signal, '_take_profit_pct', 0.50)
        sl = getattr(signal, '_stop_loss_pct', 2.0)

        self._spreads.append(SpreadPosition(
            legs=legs,
            entry_date=current_date,
            entry_cost=net_cost,
            max_profit=max_profit,
            max_loss=max_loss,
            take_profit_pct=tp,
            stop_loss_pct=sl,
            description=signal.reason,
            quantity=orders[0].quantity,
        ))

    def _calc_spread_bounds(self, legs: list[LegPosition], net_cost: float) -> tuple[float, float]:
        """Calculate max profit and max loss for a spread."""
        buy_legs = [l for l in legs if l.side == Side.BUY]
        sell_legs = [l for l in legs if l.side == Side.SELL]

        if len(legs) == 2:
            # Vertical spread (debit or credit)
            qty = legs[0].quantity
            mult = legs[0].contract.multiplier
            strikes = sorted([l.contract.strike for l in legs])
            width = (strikes[1] - strikes[0]) * qty * mult

            if net_cost > 0:
                # Debit spread: max profit = width - cost, max loss = cost
                return width - net_cost, net_cost
            else:
                # Credit spread: max profit = abs(credit), max loss = width - abs(credit)
                return abs(net_cost), width - abs(net_cost)

        elif len(legs) == 4:
            # Iron condor: max profit = net credit, max loss = wing width - credit
            qty = legs[0].quantity
            mult = legs[0].contract.multiplier
            strikes = sorted([l.contract.strike for l in legs])
            wing_width = (strikes[1] - strikes[0]) * qty * mult  # Narrower wing
            credit = abs(net_cost)
            return credit, wing_width - credit

        # Fallback
        return abs(net_cost), abs(net_cost)

    def _build_portfolio(self, current_date) -> PortfolioSnapshot:
        positions = []
        for spread in self._spreads:
            for leg in spread.legs:
                positions.append(Position(
                    contract=leg.contract,
                    quantity=leg.quantity if leg.side == Side.BUY else -leg.quantity,
                    avg_cost=leg.entry_price,
                    market_price=leg.current_price,
                ))

        equity = self._calculate_equity()
        return PortfolioSnapshot(
            timestamp=datetime.combine(current_date, datetime.min.time()) if isinstance(current_date, date) else current_date,
            net_liquidation=equity,
            cash=self._cash,
            margin_used=0,
            margin_available=self._cash,
            positions=positions,
        )

    def _calculate_equity(self) -> float:
        position_value = 0.0
        for spread in self._spreads:
            position_value += spread.current_value
        return self._cash + position_value

    def _close_all_positions(self, final_date, spot: float) -> None:
        for spread in list(self._spreads):
            self._close_spread(spread, final_date, "backtest_end")

    def _compile_results(self) -> BacktestResult:
        equity_df = pd.DataFrame(self._equity_history)
        if not equity_df.empty:
            equity_df.set_index("date", inplace=True)

        total_pnl = sum(t.pnl for t in self._trades)
        total_return = total_pnl / self.config.initial_capital if self.config.initial_capital > 0 else 0

        days = (self.config.end_date - self.config.start_date).days
        years = days / 365.0 if days > 0 else 1.0
        annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        max_dd = float(equity_df["drawdown"].max()) if not equity_df.empty else 0

        if not equity_df.empty and len(equity_df) > 1:
            daily_returns = equity_df["equity"].pct_change().dropna()
            if daily_returns.std() > 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        winning = [t for t in self._trades if t.pnl > 0]
        win_rate = len(winning) / len(self._trades) if self._trades else 0

        gross_profit = sum(t.pnl for t in self._trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self._trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_pnl = total_pnl / len(self._trades) if self._trades else 0

        return BacktestResult(
            config=self.config,
            equity_curve=equity_df,
            trades=self._trades,
            total_return=total_return,
            annualized_return=annualized,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self._trades),
            avg_trade_pnl=avg_pnl,
        )
