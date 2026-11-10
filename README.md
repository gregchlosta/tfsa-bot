# tfsa-bot

Weekly trend-following signals for a Wealthsimple TFSA, using CAD-listed ETFs only. Includes a news check and an MCP server so you can use it from the Claude app.

## How it works

- **Strategy:** every Friday, rank 13 ETFs by 1/3/6/12-month return minus cash. Hold the top 4 that beat cash, 25% each. Crypto is capped at 15% (BTC) and 10% (ETH). The rest goes to `CASH.TO`.
- **Backtest 2006–2026:** 13.8%/yr with a −29% max drawdown, vs 11.5% and −47% for the S&P 500. Without crypto: 9.1% and −23%.
- **Signals:** each ETF gets one weekly signal (UP/DOWN), with how long it has held and the price move that would flip it. A panel of other trend indicators is shown as context only.
- **Bitcoin:** a separate weekly IN/OUT signal using the same score, with flip prices (FBTC and BTC-USD), the indicator panel and a halving-cycle clock. `research` section 8 shows why the rule wasn't replaced by a faster one.
- **News check:** Claude reads recent headlines and can only reduce risk. It runs as a separate paper track until it proves itself (`performance`).

## Setup

```
uv sync
export ANTHROPIC_API_KEY=...          # optional, enables the news check (~$3/month)
uv run tfsa-bot install-desktop       # add MCP server to Claude Desktop, then restart it
uv run tfsa-bot install-schedule      # run Mon–Fri 9:45, 12:30, 16:45 with notifications
```

## Usage

```
uv run tfsa-bot brief                         # full report: trends, model portfolio, changes, Bitcoin, news
uv run tfsa-bot btc                           # Bitcoin signal, flip price, indicators, cycle clock
uv run tfsa-bot signal ZEB.TO                 # one asset: a universe ETF or any Yahoo ticker (NVDA, XEQT.TO)
uv run tfsa-bot performance                   # forward test (paper) vs buy-and-hold
uv run tfsa-bot research                      # full backtest report
uv run pytest                                 # tests
```

The bot only analyses the market and gives signals. It knows nothing about your money or how you
split it; the Claude app reads the report through MCP and works out allocation and trades with you.

Settings are in `data/config.json`.

Not financial advice.
