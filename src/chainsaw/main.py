"""Entry point for the trading platform."""

from __future__ import annotations

import asyncio

from chainsaw.broker.ibkr import IBKRBroker
from chainsaw.config import load_config
from chainsaw.data.polygon_provider import PolygonDataProvider
from chainsaw.engine import TradingEngine
from chainsaw.logging import setup_logging, get_logger
from chainsaw.pricing.engine import PricingEngine
from chainsaw.risk.manager import RiskManager
from chainsaw.strategy.iron_condor import IronCondorStrategy
from chainsaw.strategy.vertical_spread import VerticalSpreadStrategy

log = get_logger(__name__)

WATCHLIST = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]


def build_engine(config=None) -> TradingEngine:
    """Construct the full trading engine with all components."""
    if config is None:
        config = load_config()

    setup_logging(config.log_level)

    broker = IBKRBroker(config.ibkr)
    data = PolygonDataProvider(config.polygon)
    pricing = PricingEngine()
    risk = RiskManager(config.risk)

    strategies = [
        VerticalSpreadStrategy(
            symbols=WATCHLIST,
            broker=broker,
            data=data,
            pricing=pricing,
        ),
        IronCondorStrategy(
            symbols=WATCHLIST,
            broker=broker,
            data=data,
            pricing=pricing,
        ),
    ]

    return TradingEngine(
        broker=broker,
        risk_manager=risk,
        strategies=strategies,
        tick_interval=60.0,
    )


async def run() -> None:
    config = load_config()

    if config.mode != "paper":
        log.critical("refusing_live_mode", msg="Set PLATFORM_MODE=paper for safety. Live mode requires explicit override.")
        return

    engine = build_engine(config)

    try:
        await engine.start()
    except KeyboardInterrupt:
        log.info("keyboard_interrupt")
    finally:
        await engine.stop()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
