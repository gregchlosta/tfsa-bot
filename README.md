# tfsa-bot

Weekly trend-following signals for a Wealthsimple TFSA, using CAD-listed ETFs only. Includes a news check and an MCP server so you can use it from the Claude app.

## How it works

- **Strategy:** every Friday, rank 13 ETFs by 1/3/6/12-month return minus cash. Hold the top 4 that beat cash, 25% each. Crypto is capped at 15% (BTC) and 10% (ETH). The rest goes to `CASH.TO`.
- **Backtest 2006–2026:** 13.8%/yr with a −29% max drawdown, vs 11.5% and −47% for the S&P 500. Without crypto: 9.1% and −23%.
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
uv run tfsa-bot brief                         # what to hold, what to trade
uv run tfsa-bot deposit 2000                  # log money added
uv run tfsa-bot log-trade BUY ZEB 6.42 77.86  # log a fill
uv run tfsa-bot performance                   # forward test vs buy-and-hold
uv run tfsa-bot research                      # full backtest report
uv run pytest                                 # tests
```

Settings are in `data/config.json`.

Not financial advice.
