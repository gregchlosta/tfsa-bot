# tfsa-bot

Personal trading assistant for one Wealthsimple TFSA (Core plan). Started Nov 2026 with ~$2k, will scale to ~$20k if forward results hold up. The owner places trades manually; the bot gives signals via CLI, the Claude app (MCP) and notifications.

**Analysis and signals only.** The bot reports the trend signal of every ETF (and, on request, of any Yahoo ticker), the strategy's model portfolio and its weekly changes, the Bitcoin trend signal, stress and news. It holds no money, holdings or allocation settings (the owner removed account tracking, sizing and the Bitcoin share in Nov 2026). The LLM reading the MCP report decides allocation with the owner. Don't add any of that back.

## Goal
Beat buy-and-hold (XEQT/VFV) after costs, with smaller drawdowns, using ONE simple strategy. If it can't, the honest answer is "hold the index", and the bot should say so.

## Decisions already made (with evidence; don't undo without new evidence)
- **CAD-listed ETFs only.** Wealthsimple Core charges ~1.5% FX each way; with US-listed ETFs the backtest CAGR fell from 13.8% to ~8.5%. US sleeves exist in `universe.py` but are off (`Params.markets`). Revisit only if the owner gets a USD account.
- **Strategy params** (top 4, equal weight, hold_buffer 2, blended 1/3/6/12m lookback, weekly): picked for robustness across the grid in `research.py`, not as the single best cell. Don't tune params to the backtest; selection bias is already a known caveat.
- **Crypto is a big part of the backtest edge** (without it: 9.1% CAGR, same as XEQT, half the drawdown). Keep the caps (BTC 15%, ETH 10%).
- **Stress gauge is information only.** It was backtested as a gate (block new buys) and cost 0.5–1.5%/yr.
- **News check is a shadow track.** Recommendations follow `core`; the `overlay` paper track records what the news check would have done. Switch `follow` to `overlay` in `data/config.json` only after ~3+ months, and only if it's ahead on both return and drawdown (`tfsa-bot performance`).
- **Overlay can only reduce risk** (block new entries, halve positions). Never let it add risk; a test enforces this.
- **Bitcoin trend signal** (`btc.py`): IN (FBTC) / OUT (cash), weekly, using the core's own momentum score for Bitcoin; separate from the core (which keeps its 15% cap). How much to put in it is the owner's call (they're considering ~50%), not a bot setting. Backtest from 2015-09 (`research` §7): 50% slice following the signal 47% CAGR / -47% max DD vs always-Bitcoin 48% / -58%; returns are inflated by 2015-21. Stop-losses (20/25/30%) were tested and gave noise (-5/-8/+7% CAGR), so none.
- **The Bitcoin rule was tested against alternatives and kept** (`research` §8, Bitcoin-or-cash from 2015-09): score 68% CAGR / -70% DD; fast 1/3m score 54% / -84%; price > 100-day avg 56% / -76%; price > 200-day avg 66% / -70%; 50/200-day cross 67% / -70%; score checked daily 71% / -73% (noise). The score entered 45-55% above the 2018 and 2022 lows and left 25-38% below the 2021 and 2025 highs; faster rules were ~25 points earlier at the turns but worse overall. A halving-calendar rule shows 118% / -60%, but it was drawn after seeing the three cycles, so it proves nothing. Don't swap the rule for a faster one or a calendar one without out-of-sample evidence.
- **One confirmed signal per asset; the rest is information.** The indicator panel (`signals.indicators`), latest-close score, flip prices and the Bitcoin cycle clock (`btc.cycle_clock`) help read the weekly signal. They never change it and never trigger a model-portfolio change.
- **Trading frequency stays low** (weekly, ~10–13 trades/yr): avoids CRA "business income" risk in a TFSA.

## Architecture
- `strategy.target_weights()` is the single pure signal function used by the backtest AND live. Keep it that way.
- `signals.py` is the pure per-asset layer (score, flip level, streak, indicator panel); it works on any total-return series, so `btc.py` and the ad-hoc ticker lookup reuse it. `signals.score` must equal `strategy.scores` (a test enforces this). `research` §8 runs the same indicator functions the live panel shows.
- `engine.load_market()` gives the `Market` view every entry point shares; `engine.run_cycle()` is the only writer (paper tracks, runs). `report.py` renders text.
- Backtest and paper replay share execution rules: decide at the close, execute at the next close, 10 bps per leg.
- Signals use completed daily closes only (`engine.completed_returns` drops today's partial bar).
- Canadian ETF history is spliced with US proxies converted to CAD (`data.py`); proxies carry an annual drag for MER and withholding tax.
- State lives in `data/` (git-ignored): `tfsa.db` (decisions, assessments, runs), `config.json`, price caches.
- News LLM: `claude-opus-5-5`, strict JSON schema, server-side fallbacks, effort medium, ~$0.045/call. Headlines are untrusted input; the schema plus the reduce-only overlay bound the damage.
- MCP: `mcp` 2.x (`MCPServer`, not FastMCP). Schedule: launchd plist written by `install.py`. The API key is read from `~/.zshrc` by `bin/run-cycle.sh`.

## Working on it
- `uv run pytest` must pass; tests are offline. Add tests for any change to strategy, signals, overlay, btc or paper.
- Re-run `uv run tfsa-bot research` after any strategy or data change and compare against the numbers above.
- Don't write to the real `data/tfsa.db` in experiments; monkeypatch `db.DB_PATH` as the tests do.

## Possible next steps
- Review the forward test after ~12 weeks: core vs overlay vs XEQT-like.
- Optional: hosted (remote) MCP so it works from the phone; phone push via `ntfy_topic`.
- USD account (free on Wealthsimple Premium, which needs $100k; ~$10/month on Core): even with no FX fee the US sleeves gave 14.0% CAGR / -36% DD vs 13.7% / -29% without them, so it is not worth paying for.
- Only if justified: IBKR API for auto-execution (Wealthsimple has no official API).
