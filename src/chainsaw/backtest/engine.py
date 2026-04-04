"""Backtesting engine for options strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

from chainsaw.logging import get_logger
from chainsaw.models import (
    Greeks,
    OptionContract,
    OptionQuote,
    OptionType,
    Order,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    Position,
    Side,
    StockQuote,
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
    contract: str = ""
    side: str = ""
    quantity: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    commission: float = 0.0


@dataclass
class BacktestResult:
    config: BacktestConfig
    equity_curve: pd.DataFrame  # date, equity, drawdown
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


class SimulatedPosition:
    """Tracks a position during backtesting."""

    def __init__(
        self,
        contract: OptionContract,
        side: Side,
        quantity: int,
        entry_price: float,
        entry_date: date,
    ) -> None:
        self.contract = contract
        self.side = side
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.current_price = entry_price


class BacktestEngine:
    """Event-driven backtester for options strategies.

    Replays historical data day-by-day, feeding simulated portfolio snapshots
    to strategies and tracking simulated fills.
    """

    def __init__(self, config: BacktestConfig, pricing: PricingEngine | None = None) -> None:
        self.config = config
        self.pricing = pricing or PricingEngine(risk_free_rate=config.risk_free_rate)
        self._cash = config.initial_capital
        self._positions: list[SimulatedPosition] = []
        self._trades: list[TradeRecord] = []
        self._equity_history: list[dict] = []
        self._peak_equity = config.initial_capital

    def run(
        self,
        strategy: Strategy,
        price_data: pd.DataFrame,
        iv_data: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Run backtest synchronously over historical data.

        Args:
            strategy: Strategy instance to test.
            price_data: DataFrame indexed by date with columns: open, high, low, close, volume.
            iv_data: Optional DataFrame indexed by date with column: iv (annualized).
        """
        import asyncio

        dates = self._get_trading_dates(price_data)
        log.info("backtest_start", strategy=strategy.name, dates=len(dates))

        for current_date in dates:
            row = price_data.loc[current_date]
            spot = float(row["close"])
            iv = float(iv_data.loc[current_date]["iv"]) if iv_data is not None and current_date in iv_data.index else 0.20

            # Update position prices
            self._mark_positions(current_date, spot, iv)

            # Expire worthless options
            self._handle_expirations(current_date, spot)

            # Build simulated portfolio snapshot
            portfolio = self._build_portfolio(current_date)

            # Evaluate strategy
            signal: Signal = asyncio.get_event_loop().run_until_complete(strategy.evaluate(portfolio))

            # Process signal orders
            if signal.signal_type != SignalType.HOLD:
                self._process_signal(signal, current_date, spot, iv)

            # Record equity
            equity = self._calculate_equity(spot)
            self._peak_equity = max(self._peak_equity, equity)
            drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0
            self._equity_history.append({
                "date": current_date,
                "equity": equity,
                "drawdown": drawdown,
                "cash": self._cash,
                "positions_value": equity - self._cash,
            })

        # Close remaining positions at final prices
        final_row = price_data.iloc[-1]
        self._close_all_positions(dates[-1], float(final_row["close"]))

        return self._compile_results()

    def _get_trading_dates(self, price_data: pd.DataFrame) -> list:
        mask = (price_data.index >= pd.Timestamp(self.config.start_date)) & \
               (price_data.index <= pd.Timestamp(self.config.end_date))
        return list(price_data.index[mask])

    def _mark_positions(self, current_date: date, spot: float, iv: float) -> None:
        for pos in self._positions:
            result = self.pricing.price(pos.contract, spot, iv)
            pos.current_price = result.theoretical_price

    def _handle_expirations(self, current_date, spot: float) -> None:
        expired = []
        for pos in self._positions:
            exp = pos.contract.expiration
            # Handle both Timestamp and date comparison
            current = current_date.date() if hasattr(current_date, 'date') else current_date
            if exp <= current:
                # Settle at intrinsic value
                intrinsic = self.pricing._intrinsic(pos.contract, spot)
                pnl = self._calculate_position_pnl(pos, intrinsic)
                self._cash += pnl + (intrinsic * pos.quantity * pos.contract.multiplier if pos.side == Side.BUY else
                                      -intrinsic * pos.quantity * pos.contract.multiplier)
                self._trades.append(TradeRecord(
                    entry_date=pos.entry_date if isinstance(pos.entry_date, date) else pos.entry_date.date(),
                    exit_date=current,
                    contract=str(pos.contract),
                    side=pos.side.value,
                    quantity=pos.quantity,
                    entry_price=pos.entry_price,
                    exit_price=intrinsic,
                    pnl=pnl,
                    commission=self.config.commission_per_contract * pos.quantity * 2,
                ))
                expired.append(pos)

        for pos in expired:
            self._positions.remove(pos)

    def _calculate_position_pnl(self, pos: SimulatedPosition, exit_price: float) -> float:
        multiplier = pos.contract.multiplier
        if pos.side == Side.BUY:
            raw_pnl = (exit_price - pos.entry_price) * pos.quantity * multiplier
        else:
            raw_pnl = (pos.entry_price - exit_price) * pos.quantity * multiplier
        commission = self.config.commission_per_contract * pos.quantity * 2  # entry + exit
        return raw_pnl - commission

    def _process_signal(self, signal: Signal, current_date, spot: float, iv: float) -> None:
        for order in signal.orders:
            if not isinstance(order.contract, OptionContract):
                continue

            # Simulate fill with slippage
            result = self.pricing.price(order.contract, spot, iv)
            theo_price = result.theoretical_price

            if order.side == Side.BUY:
                fill_price = theo_price * (1 + self.config.slippage_pct)
                cost = fill_price * order.quantity * order.contract.multiplier
                commission = self.config.commission_per_contract * order.quantity
                self._cash -= cost + commission
            else:
                fill_price = theo_price * (1 - self.config.slippage_pct)
                credit = fill_price * order.quantity * order.contract.multiplier
                commission = self.config.commission_per_contract * order.quantity
                self._cash += credit - commission

            self._positions.append(SimulatedPosition(
                contract=order.contract,
                side=order.side,
                quantity=order.quantity,
                entry_price=fill_price,
                entry_date=current_date,
            ))

    def _build_portfolio(self, current_date) -> PortfolioSnapshot:
        positions = []
        for pos in self._positions:
            positions.append(Position(
                contract=pos.contract,
                quantity=pos.quantity if pos.side == Side.BUY else -pos.quantity,
                avg_cost=pos.entry_price,
                market_price=pos.current_price,
            ))

        equity = self._calculate_equity_from_positions()
        return PortfolioSnapshot(
            timestamp=datetime.combine(current_date, datetime.min.time()) if isinstance(current_date, date) else current_date,
            net_liquidation=equity,
            cash=self._cash,
            margin_used=0,
            margin_available=self._cash,
            positions=positions,
        )

    def _calculate_equity(self, spot: float) -> float:
        return self._calculate_equity_from_positions()

    def _calculate_equity_from_positions(self) -> float:
        position_value = 0.0
        for pos in self._positions:
            mv = pos.current_price * pos.quantity * pos.contract.multiplier
            if pos.side == Side.BUY:
                position_value += mv
            else:
                # Short positions: we received premium, now owe current value
                position_value -= mv
        return self._cash + position_value

    def _close_all_positions(self, final_date, spot: float) -> None:
        for pos in list(self._positions):
            pnl = self._calculate_position_pnl(pos, pos.current_price)
            d = final_date.date() if hasattr(final_date, 'date') else final_date
            self._trades.append(TradeRecord(
                entry_date=pos.entry_date.date() if hasattr(pos.entry_date, 'date') else pos.entry_date,
                exit_date=d,
                contract=str(pos.contract),
                side=pos.side.value,
                quantity=pos.quantity,
                entry_price=pos.entry_price,
                exit_price=pos.current_price,
                pnl=pnl,
                commission=self.config.commission_per_contract * pos.quantity * 2,
            ))
        self._positions.clear()

    def _compile_results(self) -> BacktestResult:
        equity_df = pd.DataFrame(self._equity_history)
        if not equity_df.empty:
            equity_df.set_index("date", inplace=True)

        total_pnl = sum(t.pnl for t in self._trades)
        total_return = total_pnl / self.config.initial_capital if self.config.initial_capital > 0 else 0

        # Annualized return
        days = (self.config.end_date - self.config.start_date).days
        years = days / 365.0 if days > 0 else 1.0
        annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Max drawdown
        max_dd = float(equity_df["drawdown"].max()) if not equity_df.empty else 0

        # Sharpe ratio (daily returns)
        if not equity_df.empty and len(equity_df) > 1:
            daily_returns = equity_df["equity"].pct_change().dropna()
            if daily_returns.std() > 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Win rate
        winning = [t for t in self._trades if t.pnl > 0]
        win_rate = len(winning) / len(self._trades) if self._trades else 0

        # Profit factor
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
