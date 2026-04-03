"""Main trading engine — orchestrates strategies, risk, and execution."""

from __future__ import annotations

import asyncio
from datetime import datetime

from chainsaw.broker.base import Broker
from chainsaw.logging import get_logger
from chainsaw.models import OrderStatus
from chainsaw.risk.manager import RiskManager
from chainsaw.strategy.base import Signal, SignalType, Strategy

log = get_logger(__name__)


class TradingEngine:
    """Core event loop that runs strategies, checks risk, and executes orders."""

    def __init__(
        self,
        broker: Broker,
        risk_manager: RiskManager,
        strategies: list[Strategy],
        tick_interval: float = 60.0,
    ) -> None:
        self.broker = broker
        self.risk = risk_manager
        self.strategies = strategies
        self.tick_interval = tick_interval
        self._running = False

    async def start(self) -> None:
        """Connect to broker and start the main loop."""
        log.info("engine_starting", strategies=[s.name for s in self.strategies])
        await self.broker.connect()

        # Initialize daily risk tracking
        portfolio = await self.broker.get_portfolio()
        self.risk.reset_daily(portfolio.net_liquidation)

        for strategy in self.strategies:
            await strategy.on_start()

        self._running = True
        await self._run_loop()

    async def stop(self) -> None:
        """Gracefully shut down the engine."""
        log.info("engine_stopping")
        self._running = False
        for strategy in self.strategies:
            await strategy.on_stop()
        await self.broker.disconnect()

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                log.error("tick_error", error=str(e))

            await asyncio.sleep(self.tick_interval)

    async def _tick(self) -> None:
        portfolio = await self.broker.get_portfolio()

        # Monitor portfolio risk
        risk_status = self.risk.monitor_portfolio(portfolio)
        if not risk_status.approved:
            log.critical("risk_halt", reason=risk_status.reason)
            return

        # Evaluate each strategy
        for strategy in self.strategies:
            if not strategy.enabled:
                continue

            signal: Signal = await strategy.evaluate(portfolio)

            if signal.signal_type == SignalType.HOLD:
                continue

            log.info("signal", strategy=strategy.name, type=signal.signal_type.value, reason=signal.reason)

            # Risk check each order
            for order in signal.orders:
                check = self.risk.check_order(order, portfolio)
                if not check.approved:
                    log.warning("order_rejected", strategy=strategy.name, reason=check.reason)
                    continue

                submitted = await self.broker.submit_order(order)
                log.info(
                    "order_submitted",
                    strategy=strategy.name,
                    contract=str(order.contract),
                    side=order.side.value,
                    qty=order.quantity,
                    order_id=submitted.order_id,
                )

    async def run_once(self) -> None:
        """Execute a single tick — useful for testing."""
        await self.broker.connect()
        portfolio = await self.broker.get_portfolio()
        self.risk.reset_daily(portfolio.net_liquidation)
        await self._tick()
        await self.broker.disconnect()
