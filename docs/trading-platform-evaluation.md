# Trading Platform Evaluation

## Objective

Evaluate brokers, tools, and strategies for building an autonomous algorithmic trading platform focused on high-leverage options strategies.

---

## 1. Broker & API Comparison

### Interactive Brokers (IBKR)

| Category | Details |
|---|---|
| **API** | TWS API + `ib_insync` Python wrapper |
| **Options Support** | Full chain access, complex multi-leg orders, combo orders |
| **Margin/Leverage** | Reg-T and Portfolio Margin (PM). PM can offer 6:1+ leverage on hedged positions |
| **Market Data** | Real-time and historical; included for certain tiers, otherwise $10-30/mo |
| **Commissions** | $0.65/contract (tiered can go lower at volume) |
| **Paper Trading** | Built-in paper trading environment with same API |
| **Reliability** | Battle-tested, used by institutions and serious retail |
| **Downsides** | TWS gateway can be finicky; API has a learning curve; requires TWS or IB Gateway running |
| **Best For** | Production-grade autonomous trading with full leverage |

### Tastytrade

| Category | Details |
|---|---|
| **API** | Official REST API (relatively new, actively developed) |
| **Options Support** | Excellent — platform built for options traders |
| **Margin/Leverage** | Reg-T and Portfolio Margin available |
| **Market Data** | Streaming via API |
| **Commissions** | $1.00/contract to open, $0 to close (capped at $10/leg) |
| **Paper Trading** | Available |
| **Reliability** | Growing platform, API is newer but functional |
| **Downsides** | API ecosystem less mature than IBKR; smaller community |
| **Best For** | Options-focused strategies with simpler API integration |

### Alpaca

| Category | Details |
|---|---|
| **API** | Modern REST + WebSocket, official Python SDK |
| **Options Support** | Options trading supported (added 2023-2024) |
| **Margin/Leverage** | 2x margin on stocks; options margin is standard |
| **Market Data** | Free tier available, paid plans for full depth |
| **Commissions** | $0 stocks, $0.02/contract options (may vary) |
| **Paper Trading** | Excellent paper trading environment |
| **Reliability** | Good uptime, modern infrastructure |
| **Downsides** | Options support is newer and less full-featured than IBKR; limited complex order types |
| **Best For** | Rapid prototyping, simpler strategies, getting started quickly |

### Tradier

| Category | Details |
|---|---|
| **API** | REST API, API-first broker |
| **Options Support** | Full options chain, multi-leg orders |
| **Margin/Leverage** | Standard Reg-T |
| **Market Data** | Included with funded accounts |
| **Commissions** | $0.35/contract (with subscription plan) |
| **Paper Trading** | Sandbox environment available |
| **Reliability** | Solid, used by several fintech apps as backend |
| **Downsides** | Smaller community; no portfolio margin |
| **Best For** | Cost-effective API-first options trading |

### Robinhood (Current Account)

| Category | Details |
|---|---|
| **API** | No official API. `robin_stocks` is unofficial/reverse-engineered |
| **Options Support** | Basic options (no complex multi-leg via API) |
| **Margin/Leverage** | Robinhood Gold margin (limited) |
| **Market Data** | Limited via unofficial API |
| **Commissions** | $0 (but wide spreads, PFOF) |
| **Paper Trading** | None |
| **Reliability** | Unofficial API can break at any time; account ban risk |
| **Downsides** | ToS violation risk; no portfolio margin; unreliable for automation |
| **Best For** | Manual trading only — not recommended for autonomous systems |

### Broker Verdict

```
Recommended Primary:   Interactive Brokers (IBKR)
Recommended Secondary: Tastytrade (options-specific strategies)
Prototyping/Dev:       Alpaca (fast iteration, good paper trading)
Avoid for Automation:  Robinhood (no official API, ToS risk)
```

---

## 2. Data & Analytics Stack

### Market Data Providers

| Provider | Type | Cost | Notes |
|---|---|---|---|
| **Polygon.io** | REST + WebSocket | $29-199/mo | Excellent options data, reliable |
| **IBKR Market Data** | Via broker API | $0-30/mo | Bundled if using IBKR |
| **Unusual Whales** | Options flow | $47-97/mo | Unusual activity, dark pool data |
| **CBOE LiveVol** | Options analytics | Enterprise pricing | Institutional-grade vol surfaces |
| **yfinance** | REST (Yahoo scrape) | Free | Fine for backtesting, not real-time |
| **ORATS** | Options data/analytics | $99+/mo | Historical IV, skew data |

### Options Analytics Libraries

| Library | Purpose | Notes |
|---|---|---|
| **QuantLib** (Python bindings) | Pricing, Greeks, vol surfaces | Industry standard, steep learning curve |
| **py_vollib** | Black-Scholes, Greeks | Lightweight, good for quick calculations |
| **mibian** | Options pricing | Simple, good for prototyping |
| **scipy.optimize** | Implied vol solving | Build custom models |

### Backtesting Frameworks

| Framework | Options Support | Notes |
|---|---|---|
| **vectorbt** | Limited native | Fast vectorized backtesting; extend for options |
| **backtrader** | Via plugins | Flexible, event-driven; community options extensions |
| **OptionSuite** / custom | Native | Often need to build custom for options-specific backtesting |
| **QuantConnect (Lean)** | Full | Cloud-based, good options support, free tier available |

### Recommended Stack

```
Data:         Polygon.io (primary) + IBKR (execution data)
Analytics:    QuantLib + py_vollib
Backtesting:  Custom framework on vectorbt base (options require custom logic)
Flow Data:    Unusual Whales (optional, for sentiment/flow signals)
```

---

## 3. Options Strategies — Leverage vs Risk Analysis

### Tier 1: Defined Risk, Capital Efficient

These strategies offer leverage through options mechanics while capping max loss.

#### Vertical Spreads (Bull Call / Bear Put)
- **Leverage:** 3-10x capital efficiency vs stock
- **Max Loss:** Net debit paid (defined)
- **Edge:** Profit from directional moves with less capital
- **Automation Complexity:** Low — simple entry/exit rules
- **Example:** Buy $100 call, sell $105 call for $2.00 → max profit $3.00 (150% return) vs buying 100 shares

#### Poor Man's Covered Calls (LEAPS + Short Calls)
- **Leverage:** 5-10x vs owning stock
- **Max Loss:** LEAPS premium (defined but large)
- **Edge:** Collect premium on leveraged position
- **Automation Complexity:** Medium — need to manage short call rolls
- **Example:** Buy 0.80 delta LEAPS for $20, sell monthly 0.30 delta calls for $1.50

### Tier 2: Income Strategies, Range Bound

#### Iron Condors
- **Leverage:** Capital efficient (margin on one side only)
- **Max Loss:** Width of spread minus credit (defined)
- **Edge:** Profit from time decay and range-bound markets
- **Automation Complexity:** Medium — adjustment logic needed
- **Win Rate:** High (65-80%) but poor risk/reward per trade

#### Butterflies
- **Leverage:** Very capital efficient
- **Max Loss:** Net debit (defined, small)
- **Edge:** Cheap directional or pinning bets
- **Automation Complexity:** Low — binary outcome, less management

### Tier 3: Volatility Strategies (Higher Complexity)

#### Gamma Scalping (Long Straddle + Delta Hedging)
- **Leverage:** Moderate
- **Max Loss:** Premium paid minus scalping profits
- **Edge:** Profit from realized vol > implied vol
- **Automation Complexity:** High — continuous delta hedging required
- **Best For:** Autonomous systems (frequent adjustments suit automation)

#### Volatility Arbitrage (IV vs Realized)
- **Leverage:** Moderate
- **Max Loss:** Varies by structure
- **Edge:** Statistical edge when IV misprices realized vol
- **Automation Complexity:** High — requires vol surface modeling
- **Best For:** Quantitative edge, mean-reversion of vol

### Tier 4: High Leverage / High Risk

#### Naked Short Options (Selling Puts/Calls)
- **Leverage:** High (portfolio margin amplifies)
- **Max Loss:** Theoretically unlimited (calls) or strike price (puts)
- **Edge:** Collect premium, profit from overpriced IV
- **Automation Complexity:** Medium — but risk management is critical
- **Warning:** One tail event can wipe out months of gains

#### 0DTE (Zero Days to Expiration) Strategies
- **Leverage:** Extreme
- **Max Loss:** Defined if using spreads
- **Edge:** Rapid theta decay, intraday momentum
- **Automation Complexity:** Very High — millisecond decisions, needs fast execution
- **Warning:** Extremely volatile P&L, requires robust infrastructure

### Strategy Verdict

```
Start With:        Vertical spreads + Iron condors (Tier 1-2)
Scale Into:        Gamma scalping + Vol arb (Tier 3) — plays to automation strengths
Use Cautiously:    Naked options (Tier 4) — only with strict position limits
Avoid Initially:   0DTE — infrastructure and risk requirements too high for v1
```

---

## 4. Risk Management Requirements (Non-Negotiable)

Any autonomous system MUST implement these before going live:

### Position-Level Controls
- **Max position size:** No single position > X% of portfolio
- **Max loss per trade:** Hard stop-loss or defined-risk structures
- **Greeks limits:** Max portfolio delta, gamma, vega, theta exposure

### Portfolio-Level Controls
- **Max drawdown kill switch:** Halt all trading if portfolio drops X% in a day/week
- **Correlation limits:** Don't over-concentrate in correlated underlyings
- **Margin utilization cap:** Never use > 70% of available margin
- **Sector exposure limits:** Cap exposure per sector/industry

### System-Level Controls
- **Circuit breaker:** Pause trading on API errors, data gaps, or anomalies
- **Order validation:** Sanity check all orders before submission (price, size, direction)
- **Heartbeat monitoring:** Alert if system goes unresponsive
- **Audit logging:** Every decision, order, and fill must be logged

---

## 5. Recommended Architecture

```
┌─────────────────────────────────────────────────────┐
│                   TRADING PLATFORM                   │
├──────────┬──────────┬───────────┬───────────────────┤
│  Data    │ Strategy │ Execution │ Risk Management   │
│  Layer   │ Engine   │ Engine    │ Module            │
├──────────┼──────────┼───────────┼───────────────────┤
│ Polygon  │ Signal   │ IBKR API  │ Position limits   │
│ IBKR     │ Generate │ Order     │ Greeks monitor    │
│ Options  │ Backtest │ Route     │ Kill switches     │
│ Chain    │ Optimize │ Fill Mgmt │ Drawdown track    │
└──────────┴──────────┴───────────┴───────────────────┘
         │              │               │
         └──────────────┴───────────────┘
                        │
              ┌─────────┴─────────┐
              │   Logging &       │
              │   Monitoring      │
              │   Dashboard       │
              └───────────────────┘
```

### Tech Stack

```
Language:       Python 3.11+
Broker API:     ib_insync (IBKR)
Data:           polygon-api-client, pandas
Analytics:      QuantLib, numpy, scipy
Scheduling:     APScheduler or asyncio event loop
Database:       SQLite (dev) → PostgreSQL (prod)
Monitoring:     Prometheus + Grafana or simple logging + alerts
Deployment:     Docker container on cloud VM (low-latency region)
```

---

## 6. Development Phases

### Phase 1 — Foundation
- [ ] Broker API integration (IBKR paper trading)
- [ ] Market data pipeline (options chains, quotes)
- [ ] Options pricing/Greeks engine
- [ ] Basic logging and monitoring

### Phase 2 — Strategy Engine
- [ ] Backtesting framework for options strategies
- [ ] Implement Tier 1 strategies (vertical spreads)
- [ ] Signal generation module
- [ ] Paper trading loop

### Phase 3 — Risk & Execution
- [ ] Risk management module (all controls from Section 4)
- [ ] Smart order routing and fill management
- [ ] Portfolio-level Greeks dashboard
- [ ] Kill switch implementation

### Phase 4 — Live Trading
- [ ] Migrate from paper to live (small capital)
- [ ] Monitoring and alerting
- [ ] Performance analytics and reporting
- [ ] Gradual scale-up with Tier 2-3 strategies

---

## 7. Decision Summary

| Decision | Choice | Rationale |
|---|---|---|
| **Primary Broker** | Interactive Brokers | Best API, portfolio margin, institutional-grade |
| **Dev/Prototype Broker** | Alpaca | Fast iteration, modern API |
| **Data Provider** | Polygon.io + IBKR | Reliable options data, reasonable cost |
| **Pricing Library** | QuantLib + py_vollib | Industry standard + lightweight fallback |
| **Initial Strategies** | Vertical spreads, Iron condors | Defined risk, automatable, capital efficient |
| **Target Strategies** | Gamma scalping, Vol arb | Automation advantage, quantitative edge |
| **Robinhood** | Manual use only | No official API, not suitable for automation |
