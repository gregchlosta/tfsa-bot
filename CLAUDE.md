# tfsa-bot

Personal trading assistant for one Wealthsimple TFSA (Core plan). Started Nov 2026 with ~$2k, will scale to ~$20k if forward results hold up. The owner places trades manually; the bot gives signals via CLI, the Claude app (MCP) and notifications.

## Goal
Beat buy-and-hold (XEQT/VFV) after costs, with smaller drawdowns, using ONE simple strategy. If it can't, the honest answer is "hold the index", and the bot should say so.

## Decisions already made (with evidence; don't undo without new evidence)
- **CAD-listed ETFs only.** Wealthsimple Core charges ~1.5% FX each way; with US-listed ETFs the backtest CAGR fell from 13.8% to ~8.5%. US sleeves exist in `universe.py` but are off (`Params.markets`). Revisit only if the owner gets a USD account.
- **Strategy params** (top 4, equal weight, hold_buffer 2, blended 1/3/6/12m lookback, weekly): picked for robustness across the grid in `research.py`, not as the single best cell. Don't tune params to the backtest; selection bias is already a known caveat.
- **Crypto is a big part of the backtest edge** (without it: 9.1% CAGR, same as XEQT, half the drawdown). Keep the caps (BTC 15%, ETH 10%).
- **Stress gauge is information only.** It was backtested as a gate (block new buys) and cost 0.5–1.5%/yr.
- **News check is a shadow track.** Recommendations follow `core`; the `overlay` paper track records what the news check would have done. Switch `follow` to `overlay` in `data/config.json` only after ~3+ months, and only if it's ahead on both return and drawdown (`tfsa-bot performance`).
- **Overlay can only reduce risk** (block new entries, halve positions). Never let it add risk; a test enforces this.
- **Bitcoin pot** (`btc.py`, `btc_share` in config; owner chose 50% in Nov 2026): that share is all FBTC or all CASH.TO, flipped weekly on the core's own momentum score for Bitcoin. The core underneath is unchanged (keeps its 15% cap). Backtest from 2015-09 (`research` §7): 50% pot with the rule 47% CAGR / -47% max DD vs always-Bitcoin 48% / -58%; returns are inflated by 2015-21. Stop-losses (20/25/30%) were tested and gave noise (-5/-8/+7% CAGR), so none. Mid-week monitoring (live score, flip price) is information only.
- **Trading frequency stays low** (weekly, ~10–13 trades/yr): avoids CRA "business income" risk in a TFSA.

## Architecture
- `strategy.target_weights()` is the single pure signal function used by the backtest AND live. Keep it that way.
- Backtest and paper replay share execution rules: decide at the close, execute at the next close, 10 bps per leg.
- Signals use completed daily closes only (`engine.completed_returns` drops today's partial bar).
- Canadian ETF history is spliced with US proxies converted to CAD (`data.py`); proxies carry an annual drag for MER and withholding tax.
- State lives in `data/` (git-ignored): `tfsa.db` (decisions, assessments, trades, deposits, runs), `config.json`, price caches.
- News LLM: `claude-opus-5-5`, strict JSON schema, server-side fallbacks, effort medium, ~$0.045/call. Headlines are untrusted input; the schema plus the reduce-only overlay bound the damage.
- MCP: `mcp` 2.x (`MCPServer`, not FastMCP). Schedule: launchd plist written by `install.py`. The API key is read from `~/.zshrc` by `bin/run-cycle.sh`.

## Working on it
- `uv run pytest` must pass; tests are offline. Add tests for any change to strategy, overlay, portfolio or paper.
- Re-run `uv run tfsa-bot research` after any strategy or data change and compare against the numbers above.
- Don't write to the real `data/tfsa.db` in experiments; monkeypatch `db.DB_PATH` as the tests do.

## Possible next steps
- Review the forward test after ~12 weeks: core vs overlay vs XEQT-like.
- Optional: hosted (remote) MCP so it works from the phone; phone push via `ntfy_topic`.
- Only if justified: USD account → enable US sleeves; IBKR API for auto-execution (Wealthsimple has no official API).
