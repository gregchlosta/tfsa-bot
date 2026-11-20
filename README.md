# tfsa-bot

Weekly buy and sell signals for a Wealthsimple TFSA. It runs one simple, backtested trend-following strategy on 13 CAD-listed ETFs and scans about 75 more CAD-listed names with the same signal. You get the results through the CLI, the Claude app (MCP) and notifications.

**Analysis and signals only.** The bot holds no money and knows nothing about your positions or how you split them. You place trades yourself, and the Claude app works out allocation with you from the report.

The goal is to beat buy-and-hold (XEQT/VFV) after costs, with smaller drawdowns. If it can't, the honest answer is "hold the index".

## What it gives you

| | What it is | Status |
|---|---|---|
| **Model portfolio** | Top 4 of the 13 strategy ETFs, 25% each, with BUY/SELL lines when it changes | Backtested |
| **Trend signal per asset** | UP/DOWN each Friday, how long it has held, the price move that would flip it | Backtested on the 13 ETFs |
| **Scan** | The same signal for ~89 assets, with this week's flips | Information only |
| **Bitcoin signal** | IN (hold FBTC) / OUT (cash), weekly | Backtested from 2015 |
| **Stress gauge** | Price-based market stress (VIX, credit, drawdowns) | Information only |
| **News check** | Claude reads headlines and can only reduce risk | Shadow paper track |

## The strategy

Every Friday after the close:

1. Score each ETF: its average 1/3/6/12-month total return, minus what cash earned over the same windows. A score above 0 means UP (it beats a savings account); below 0 means DOWN.
2. Hold the 4 highest-scoring ETFs that are UP, 25% each. Empty slots sit in `CASH.TO`.
3. Keep a holding while it still ranks in the top 6 (the hold buffer), to avoid churn. The brief says when a holding is "kept by the buffer" and which ETF is "next in line".
4. Cap crypto: Bitcoin at 15%, Ether at 10%.
5. Trade at the next close. That comes to about 13 trades a year, which keeps the TFSA clear of CRA "business income" risk.

**The 13 ETFs:** XIC (Canada), VFV (S&P 500), QQC (Nasdaq-100), XEF (international), XEC (emerging markets), ZEB (Canadian banks), XEG (Canadian energy), XIT (Canadian tech), ZRE (Canadian REITs), ZAG (Canadian bonds), CGL.C (gold), FBTC (Bitcoin), ETHX.B (Ether), all `.TO`.

### Backtest (2006 to Nov 2026; 10 bps per trade; trades a day after the signal)

| | CAGR | Max drawdown | 2006–15 | 2016–now |
|---|---|---|---|---|
| **Strategy** | **13.6%** | **−29%** | 8.3% | 19.2% |
| Strategy without crypto | 9.0% | −23% | | |
| XEQT-like (100% equity) | 9.2% | −46% | 5.8% | 12.8% |
| VFV (S&P 500) | 11.4% | −47% | 8.2% | 14.9% |

Read these numbers with the caveats:

- **Crypto is much of the edge.** Without it the strategy matches XEQT on return, with half the drawdown.
- **The early history is partly simulated.** Most of these ETFs are young, so their early history uses US proxies converted to CAD, less an annual drag for fees and withholding tax.
- **Parameters were picked after looking at results.** They were chosen for robustness across a grid (`research` section 2), not as the best single cell, but some selection bias remains.

### Why it looks the way it does

| Decision | Evidence |
|---|---|
| **CAD-listed only** | Wealthsimple charges ~1.5% FX each way; US-listed ETFs cut the backtest CAGR to 11.8% |
| **Fixed top 4, not every uptrend** | Holding every uptrend gave 13.3% with 29 trades a year, for no extra return (`research` §10) |
| **No stress gate** | Blocking buys when stress is high cost 0.5–1.5% a year |
| **Weekly, slow signal** | Faster rules (moving averages, 1/3-month scores, daily checks) turned earlier but did worse overall (`research` §8) |

## Reading a signal

- **UP / DOWN**: the confirmed weekly signal. It only changes on the week's last close.
- **BUY / SELL**: the signal turned UP / DOWN at the last Friday close.
- **Pending** (`>UP?` / `>DOWN?`): the latest close disagrees with the weekly signal. It is not confirmed until Friday.
- **Near**: the move that would flip it is within one typical week's move.
- **flip**: how far the price has to move from the latest close to flip the signal. Single-asset reports also give it as a price.
- **n/5**: how many of five trend indicators point up. This is context; only the first indicator (the score) is the signal.
- **record**: what following this signal on this asset alone did, next to holding it. The format is CAGR / max drawdown.

> **A BUY/SELL on one asset is a risk signal, not a return signal.** Following a single ETF's signal roughly halved its drawdown, but returned less than holding it on 12 of 13 (`research` §9; Bitcoin is the exception). For example, VFV returned 9.9% with a −21% drawdown following the signal, against 11.4% and −47% holding it. On single stocks the signal was often worse on both counts. The tested edge comes from rotating into the strongest few, which is what the model portfolio does.

## The scan

The scan covers the 13 model ETFs plus the watchlist in [universe.py](src/tfsa_bot/universe.py), which has four groups:

- **More ETFs:** broad and regional (XEQT, XQQ, XSU, Japan, India, China, …), sectors and themes (semiconductors, uranium, cybersecurity, health care, utilities, gold miners, …), bonds, silver, oil and Solana.
- **Large Canadian stocks:** banks, Shopify, Constellation, rails, energy, and more.
- **CDRs of US stocks** (Nvidia, Apple, Microsoft, … on Cboe Canada, CAD-hedged, no FX fee): Yahoo has no history for CDR tickers, so the signal uses the US stock in USD less 0.6% a year. Flip prices for CDRs are the US stock's price.
- **Your own additions**, from `tfsa-bot watch`.

The watchlist never changes the model portfolio, and only the 13 ETFs were backtested as a portfolio. If Yahoo stops updating a ticker, the report flags it as `DATA WARNING` / `STALE` instead of showing a flat price.

## Bitcoin

Bitcoin gets a separate IN/OUT signal from the same score. It comes with flip prices (FBTC and BTC-USD), the indicator panel and a halving-cycle clock; the clock is three past cycles, a description rather than evidence. Choosing how much to put in Bitcoin is up to you, not a bot setting.

Bitcoin or cash from 2015-09:

| Rule | CAGR | Max drawdown |
|---|---|---|
| Follow the signal | 68% | −70% |
| Always hold Bitcoin | 66% | −82% |

The 2015–21 gains dominate these numbers. Stop-losses were tested and added only noise.

## Setup

```
uv sync
export ANTHROPIC_API_KEY=...          # optional, in ~/.zshrc: enables the news check (~$0.05 a call)
uv run tfsa-bot install-desktop       # register the MCP server in Claude Desktop, then restart it
uv run tfsa-bot install-schedule      # launchd job: Mon–Fri 9:45, 12:30, 16:45 (log: data/run.log)
```

Each scheduled run refreshes prices, runs the news check, records the weekly decision after Friday's close, and notifies you about:

- a new model-portfolio change
- this week's trend flips across the scan
- a Bitcoin flip, or a Bitcoin heads-up when the latest close disagrees with the weekly signal
- a rise in market risk, or new news flags

Notifications are macOS banners. Set `ntfy_topic` in `data/config.json` to a long random string and subscribe to it in the ntfy app to get them on your phone too.

## Usage

```
uv run tfsa-bot brief                  # full report: model portfolio, changes, 13 ETF trends, scan events, Bitcoin, stress, news
uv run tfsa-bot opportunities          # actionable setups: model actions, fresh flips, pending flips, pullbacks, quality leaders, early turnarounds
uv run tfsa-bot compare VFV QQC NVDA   # side-by-side comparison of any tickers (model ETFs, watchlist, CDRs, or Yahoo)
uv run tfsa-bot scan                   # every scanned asset, best trend first, with this week's flips
uv run tfsa-bot scan --show events     # only rows where something is happening (also --show up|down, --kind ..., --tier model|watch, --sort score|risk_adj|flip)
uv run tfsa-bot signal ZEB             # one asset: a model ETF, a watchlist name (CHPS, SHOP, NVDA = its CDR) or any Yahoo ticker
uv run tfsa-bot btc                    # Bitcoin signal, flip prices, indicators, cycle clock
uv run tfsa-bot watch BCE.TO           # add tickers to the scan; `unwatch` removes them (default entries get hidden)
uv run tfsa-bot performance            # forward test: core vs core+news vs XEQT-like, Bitcoin signal vs holding
uv run tfsa-bot research               # full backtest report (sections 1–10)
uv run tfsa-bot run [--no-llm] [--quiet]  # one full cycle, what the scheduler runs
uv run pytest                          # offline tests
```

### In the Claude app (MCP tools)

| Tool | What it does |
|---|---|
| `get_brief` | The full report; `refresh_news=True` runs a fresh news check |
| `find_opportunities` | Actionable buy/sell setups: model actions, fresh/pending flips, pullbacks in uptrends, quality leaders, early turnarounds, extended warnings |
| `compare_assets` | Side-by-side comparison of any list of tickers in one call |
| `scan` | The scan table, filterable by `kind`, `show`, `tier`, and sortable by `score`, `risk_adj`, or `flip` |
| `get_signal` | One asset in detail, with RSI(14), risk-adjusted score, and its track record |
| `bitcoin_status` | Bitcoin in detail (read-only, fast) |
| `market_news` | Raw headlines plus the stress gauge, no AI judgement |
| `performance` | The forward test (`core`, `overlay`, `quality`, `btc`, `btc_onchain` vs benchmarks) |
| `run_backtest` | Backtest with a chosen `top_n`, `include_crypto`, `rebalance`, `ranking` (`raw`/`blend`), `hold_buffer`, and dates |
| `update_settings` | Change `follow` (`core`/`overlay`/`quality`), `use_llm`, `ntfy_topic`, and add/remove scan tickers (`watch` / `unwatch`) |

Try asking: *"What flipped this week?"*, *"Show me all uptrending sectors"*, *"Is NVDA a buy?"*, *"How has following the signal on gold worked historically?"*

## Settings and data

Everything is stored locally in `data/` (git-ignored):

- `config.json`: the settings
  - `follow`: `core` (the default), or `overlay` to follow the news-checked track. Switch only after about 3 months, and only if `performance` shows it ahead on both return and drawdown.
  - `use_llm`: turns the news check on or off.
  - `ntfy_topic`: phone push notifications.
  - `watchlist_extra` and `watchlist_hidden`: your changes to the scan.
- `tfsa.db`: the weekly decisions per paper track, news assessments, the run log and the notification state.
- `raw_prices.pkl`, `watch_prices.pkl`, `cad_tbill.pkl`: price caches (Yahoo Finance; the cash rate comes from FRED). They refresh after the TSX close or after 6 hours.

Signals use completed daily closes only; today's bar is ignored until 16:30 Toronto time.

## Code map

```
src/tfsa_bot/
  strategy.py   target_weights(): the single signal function, shared by the backtest and live runs
  signals.py    per-asset score, flip price, streak, flags (BUY/SELL/pending/near), track record, indicators
  scan.py       model ETFs + watchlist into one table with events (read-only)
  btc.py        Bitcoin IN/OUT and the cycle clock
  universe.py   the 13 strategy ETFs and the watchlist
  engine.py     load_market(), run_cycle() (the only writer), notifications
  report.py     text rendering
  data.py       downloads, CAD conversion, proxy splicing, stale-price check
  backtest.py   day-by-day simulation; research.py = the full report
  overlay.py    the news check's reduce-only rules; news.py = headlines + Claude assessment
  paper.py      forward-test replay; stress.py = stress gauge
  mcp_server.py, install.py, notify.py, db.py
```

Design rules and the evidence behind each decision are in [CLAUDE.md](CLAUDE.md).

Not financial advice.
