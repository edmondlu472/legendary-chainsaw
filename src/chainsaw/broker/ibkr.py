"""Interactive Brokers integration via ib_insync."""

from __future__ import annotations

from datetime import date, datetime

from ib_insync import IB, Contract, Option, Stock, MarketOrder, LimitOrder, StopOrder, Trade

from chainsaw.broker.base import Broker
from chainsaw.config import IBKRConfig
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

log = get_logger(__name__)


def _to_ib_option(contract: OptionContract) -> Option:
    right = "C" if contract.option_type == OptionType.CALL else "P"
    return Option(
        symbol=contract.symbol,
        lastTradeDateOrContractMonth=contract.expiration.strftime("%Y%m%d"),
        strike=contract.strike,
        right=right,
        exchange="SMART",
        multiplier=str(contract.multiplier),
    )


def _parse_order_status(status: str) -> OrderStatus:
    mapping = {
        "PendingSubmit": OrderStatus.PENDING,
        "PendingCancel": OrderStatus.PENDING,
        "PreSubmitted": OrderStatus.SUBMITTED,
        "Submitted": OrderStatus.SUBMITTED,
        "Cancelled": OrderStatus.CANCELLED,
        "Filled": OrderStatus.FILLED,
        "Inactive": OrderStatus.REJECTED,
    }
    return mapping.get(status, OrderStatus.PENDING)


class IBKRBroker(Broker):
    """Interactive Brokers implementation using ib_insync."""

    def __init__(self, config: IBKRConfig | None = None) -> None:
        self._config = config or IBKRConfig()
        self._ib = IB()

    async def connect(self) -> None:
        await self._ib.connectAsync(
            host=self._config.host,
            port=self._config.port,
            clientId=self._config.client_id,
            timeout=self._config.timeout,
            readonly=self._config.readonly,
        )
        log.info("connected_to_ibkr", host=self._config.host, port=self._config.port)

    async def disconnect(self) -> None:
        self._ib.disconnect()
        log.info("disconnected_from_ibkr")

    async def get_stock_quote(self, symbol: str) -> StockQuote:
        contract = Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)
        ticker = self._ib.reqMktData(contract, snapshot=True)
        self._ib.sleep(2)  # Wait for data
        return StockQuote(
            symbol=symbol,
            bid=ticker.bid or 0.0,
            ask=ticker.ask or 0.0,
            last=ticker.last or 0.0,
            volume=int(ticker.volume or 0),
            timestamp=datetime.now(),
        )

    async def get_option_chain(
        self,
        symbol: str,
        expiration: date | None = None,
    ) -> list[OptionQuote]:
        stock = Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(stock)
        chains = self._ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)

        if not chains:
            return []

        chain = chains[0]
        expirations = [expiration.strftime("%Y%m%d")] if expiration else list(chain.expirations)[:3]
        strikes = sorted(chain.strikes)

        quotes: list[OptionQuote] = []
        for exp in expirations:
            for strike in strikes:
                for right in ("C", "P"):
                    opt = Option(symbol, exp, strike, right, "SMART")
                    try:
                        self._ib.qualifyContracts(opt)
                        ticker = self._ib.reqMktData(opt, snapshot=True)
                        self._ib.sleep(0.5)

                        opt_type = OptionType.CALL if right == "C" else OptionType.PUT
                        contract = OptionContract(
                            symbol=symbol,
                            expiration=date.fromisoformat(f"{exp[:4]}-{exp[4:6]}-{exp[6:]}"),
                            strike=strike,
                            option_type=opt_type,
                        )
                        greeks = Greeks()
                        if ticker.modelGreeks:
                            g = ticker.modelGreeks
                            greeks = Greeks(
                                delta=g.delta or 0.0,
                                gamma=g.gamma or 0.0,
                                theta=g.theta or 0.0,
                                vega=g.vega or 0.0,
                                iv=g.impliedVol or 0.0,
                            )

                        quotes.append(OptionQuote(
                            contract=contract,
                            bid=ticker.bid or 0.0,
                            ask=ticker.ask or 0.0,
                            last=ticker.last or 0.0,
                            volume=int(ticker.volume or 0),
                            open_interest=int(ticker.open_interest or 0) if hasattr(ticker, "open_interest") else 0,
                            greeks=greeks,
                        ))
                    except Exception as e:
                        log.debug("skip_option", strike=strike, right=right, error=str(e))

        return quotes

    async def get_option_quote(self, contract: OptionContract) -> OptionQuote:
        ib_opt = _to_ib_option(contract)
        self._ib.qualifyContracts(ib_opt)
        ticker = self._ib.reqMktData(ib_opt, snapshot=True)
        self._ib.sleep(2)

        greeks = Greeks()
        if ticker.modelGreeks:
            g = ticker.modelGreeks
            greeks = Greeks(
                delta=g.delta or 0.0,
                gamma=g.gamma or 0.0,
                theta=g.theta or 0.0,
                vega=g.vega or 0.0,
                iv=g.impliedVol or 0.0,
            )

        return OptionQuote(
            contract=contract,
            bid=ticker.bid or 0.0,
            ask=ticker.ask or 0.0,
            last=ticker.last or 0.0,
            volume=int(ticker.volume or 0),
            open_interest=0,
            greeks=greeks,
        )

    async def submit_order(self, order: Order) -> Order:
        if isinstance(order.contract, OptionContract):
            ib_contract = _to_ib_option(order.contract)
        else:
            ib_contract = Stock(order.contract, "SMART", "USD")

        self._ib.qualifyContracts(ib_contract)

        action = "BUY" if order.side == Side.BUY else "SELL"

        if order.order_type == OrderType.MARKET:
            ib_order = MarketOrder(action, order.quantity)
        elif order.order_type == OrderType.LIMIT:
            ib_order = LimitOrder(action, order.quantity, order.limit_price)
        elif order.order_type == OrderType.STOP:
            ib_order = StopOrder(action, order.quantity, order.stop_price)
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

        trade: Trade = self._ib.placeOrder(ib_contract, ib_order)
        log.info("order_submitted", contract=str(order.contract), side=action, qty=order.quantity)

        order.order_id = str(trade.order.orderId)
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()
        return order

    async def cancel_order(self, order_id: str) -> bool:
        for trade in self._ib.openTrades():
            if str(trade.order.orderId) == order_id:
                self._ib.cancelOrder(trade.order)
                log.info("order_cancelled", order_id=order_id)
                return True
        return False

    async def get_order_status(self, order_id: str) -> Order:
        for trade in self._ib.trades():
            if str(trade.order.orderId) == order_id:
                return Order(
                    contract=trade.contract.symbol,
                    side=Side.BUY if trade.order.action == "BUY" else Side.SELL,
                    quantity=int(trade.order.totalQuantity),
                    order_type=OrderType.MARKET,
                    status=_parse_order_status(trade.orderStatus.status),
                    order_id=order_id,
                    fill_price=trade.orderStatus.avgFillPrice or None,
                    filled_quantity=int(trade.orderStatus.filled),
                )
        raise ValueError(f"Order {order_id} not found")

    async def get_positions(self) -> list[Position]:
        ib_positions = self._ib.positions()
        positions: list[Position] = []
        for pos in ib_positions:
            c = pos.contract
            if isinstance(c, Option):
                opt_type = OptionType.CALL if c.right == "C" else OptionType.PUT
                exp = c.lastTradeDateOrContractMonth
                contract: OptionContract | str = OptionContract(
                    symbol=c.symbol,
                    expiration=date.fromisoformat(f"{exp[:4]}-{exp[4:6]}-{exp[6:]}"),
                    strike=c.strike,
                    option_type=opt_type,
                )
            else:
                contract = c.symbol

            positions.append(Position(
                contract=contract,
                quantity=int(pos.position),
                avg_cost=pos.avgCost,
            ))
        return positions

    async def get_portfolio(self) -> PortfolioSnapshot:
        account_values = {v.tag: float(v.value) for v in self._ib.accountValues() if v.currency == "USD"}
        positions = await self.get_positions()

        return PortfolioSnapshot(
            timestamp=datetime.now(),
            net_liquidation=account_values.get("NetLiquidation", 0.0),
            cash=account_values.get("TotalCashValue", 0.0),
            margin_used=account_values.get("MaintMarginReq", 0.0),
            margin_available=account_values.get("AvailableFunds", 0.0),
            positions=positions,
        )

    async def get_expirations(self, symbol: str) -> list[date]:
        stock = Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(stock)
        chains = self._ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
        if not chains:
            return []

        exps: list[date] = []
        for exp_str in sorted(chains[0].expirations):
            exps.append(date.fromisoformat(f"{exp_str[:4]}-{exp_str[4:6]}-{exp_str[6:]}"))
        return exps
