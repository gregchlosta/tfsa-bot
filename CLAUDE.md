# tfsa-bot

Personal trading assistant for one Wealthsimple TFSA (Core plan). Started Nov 2026 with ~$2k, will scale to ~$20k if forward results hold up. The owner places trades manually; the bot gives signals via CLI, the Claude app (MCP) and notifications.

**Analysis and signals only.** The bot reports the trend signal of every ETF (and, on request, of any Yahoo ticker), a scan of a wider watchlist with this week's signal events, the strategy's model portfolio and its weekly changes, the Bitcoin trend signal, stress and news. It holds no money, holdings or allocation settings (the owner removed account tracking, sizing and the Bitcoin share in Nov 2026). The LLM reading the MCP report decides allocation with the owner. Don't add any of that back.

## Goal
Beat buy-and-hold (XEQT/VFV) after costs, with smaller drawdowns, using ONE simple strategy. If it can't, the honest answer is "hold the index", and the bot should say so.

## Decisions already made (with evidence; don't undo without new evidence)
- **CAD-listed ETFs only.** Wealthsimple Core charges ~1.5% FX each way; with US-listed ETFs the backtest CAGR fell from 13.6% to 11.8% and the max drawdown deepened from -29% to -36% (`research` §1, Nov 2026; an earlier version showed ~8.5%). US sleeves exist in `universe.py` but are off (`Params.markets`). Revisit only if the owner gets a USD account.
- **Strategy params** (top 4, equal weight, hold_buffer 2, blended 1/3/6/12m lookback, weekly): picked for robustness across the grid in `research.py`, not as the single best cell. Don't tune params to the backtest; selection bias is already a known caveat.
- **Crypto is a big part of the backtest edge** (without it: 9.0% CAGR / -23% DD vs XEQT-like 9.2% / -46%, i.e. same return, half the drawdown). Keep the caps (BTC 15%, ETH 10%).
- **Stress gauge is information only.** It was backtested as a gate (block new buys) and cost 0.5–1.5%/yr.
- **News check is a shadow track.** Recommendations follow `core`; the `overlay` paper track records what the news check would have done. Switch `follow` to `overlay` in `data/config.json` only after ~3+ months, and only if it's ahead on both return and drawdown (`tfsa-bot performance`).
- **Overlay can only reduce risk** (block new entries, halve positions). Never let it add risk; a test enforces this.
- **Bitcoin trend signal** (`btc.py`): IN (FBTC) / OUT (cash), weekly, using the core's own momentum score for Bitcoin; separate from the core (which keeps its 15% cap). How much to put in it is the owner's call (they're considering ~50%), not a bot setting. Backtest from 2015-09 (`research` §7): 50% slice following the signal 47% CAGR / -47% max DD vs always-Bitcoin 48% / -58%; returns are inflated by 2015-21. Stop-losses (20/25/30%) were tested and gave noise (-5/-8/+7% CAGR), so none.
- **The Bitcoin rule was tested against alternatives and kept** (`research` §8, Bitcoin-or-cash from 2015-09): score 68% CAGR / -70% DD; fast 1/3m score 54% / -84%; price > 100-day avg 56% / -76%; price > 200-day avg 66% / -70%; 50/200-day cross 67% / -70%; score checked daily 71% / -73% (noise). The score entered 45-55% above the 2018 and 2022 lows and left 25-38% below the 2021 and 2025 highs; faster rules were ~25 points earlier at the turns but worse overall. A halving-calendar rule shows 118% / -60%, but it was drawn after seeing the three cycles, so it proves nothing. Don't swap the rule for a faster one or a calendar one without out-of-sample evidence.
- **One confirmed signal per asset; the rest is information.** The indicator panel (`signals.indicators`), latest-close score, flip prices and the Bitcoin cycle clock (`btc.cycle_clock`) help read the weekly signal. They never change it and never trigger a model-portfolio change.
- **The scan is information only** (added Nov 2026; the owner wants every signal shown, not a fixed number of picks). `universe.WATCHLIST` plus `watchlist_extra` in the config get the same weekly signal as the 13 ETFs: more CAD-listed ETFs, large Canadian stocks, and CDRs of US stocks. Only the 13 sleeves were backtested as a portfolio; the watchlist never reaches `strategy.target_weights` (a test enforces this). Promote a watchlist ETF into `SLEEVES` only after backtesting the wider universe as a portfolio.
- **A BUY/SELL on one asset is a risk signal, not a return signal** (`research` §9, the asset while UP and cash while DOWN vs holding it). Of the 13 ETFs only Bitcoin did better following its signal (68.2% vs 65.6% CAGR); the other 12 returned less (XIC 4.6% vs 8.6%, VFV 9.9% vs 11.4%, gold 6.7% vs 10.6%) while the drawdown was cut on 12 of 13 (VFV -21% vs -47%, ZEB -24% vs -57%). On single stocks it was often worse on both counts (the Visa CDR: 8% / -59% vs 22% / -36%). So every signal is shown with its track record (`signals.track_record`); don't present a watchlist flip as a trade instruction.
- **The fixed top 4 was kept after testing no fixed count** (`research` §10, 2006 to 2026-11): top 4 13.6% CAGR / -29% DD / 12.7 trades a year; every uptrend at equal weight 13.3% / -34% / 28.7; every uptrend at 1/13 each 10.5% / -19% / 28.6; top 8 13.8% / -27% / 18.3 but 6.8% vs 8.3% in 2006-15. More trades, no more return.
- **Signal events are derived, not stored.** BUY/SELL = the weekly state changed at the last week-end (`signals.flags`); pending = the latest close disagrees; near = the flip is within one typical week's move (63-day volatility). They are recomputed from prices each run, so there is no per-asset state to keep.
- **CDRs are measured on the US stock in USD less 0.6%/yr** (`universe.CDR_DRAG`): a CAD-hedged CDR tracks the stock's USD return, and Yahoo has a single day of history for `.NE` CDR tickers. Flip prices for CDRs are the US stock's price.
- **Trading frequency stays low** (weekly, ~10–13 trades/yr): avoids CRA "business income" risk in a TFSA.

## Architecture
- `strategy.target_weights()` is the single pure signal function used by the backtest AND live. Keep it that way.
- `signals.py` is the pure per-asset layer (score, flip level, streak, indicator panel); it works on any total-return series, so `btc.py` and the ad-hoc ticker lookup reuse it. `signals.score` must equal `strategy.scores` (a test enforces this). `research` §8 runs the same indicator functions the live panel shows.
- `engine.load_market()` gives the `Market` view every entry point shares; `engine.run_cycle()` is the only writer (paper tracks, runs). `report.py` renders text.
- `scan.py` builds the scan (model ETFs + watchlist) from a `Market`; it is read-only and a failed watchlist download only costs the events section of the brief. Watchlist prices are cached in `data/watch_prices.pkl`. The brief's run summary keeps the scan's events and counts, not its rows.
- Notifications are sent by scheduled runs only. What has been announced is kept in the `state` table (`notified_week`, `btc_heads_up`), not inferred from the run log, because reports opened from the app or the CLI log runs too.
- `data.stale` flags tickers whose last price is 2+ trading days old (prices are forward-filled for 5 days, so a dead ticker would otherwise look flat). One day of lag is normal: US and Canadian holidays differ.
- Backtest and paper replay share execution rules: decide at the close, execute at the next close, 10 bps per leg.
- Signals use completed daily closes only (`engine.completed_returns` drops today's partial bar).
- Canadian ETF history is spliced with US proxies converted to CAD (`data.py`); proxies carry an annual drag for MER and withholding tax.
- State lives in `data/` (git-ignored): `tfsa.db` (decisions, assessments, runs), `config.json`, price caches.
- News LLM: `claude-opus-5-5`, strict JSON schema, server-side fallbacks, effort medium, ~$0.045/call. Headlines are untrusted input; the schema plus the reduce-only overlay bound the damage.
- MCP: `mcp` 2.x (`MCPServer`, not FastMCP). Schedule: launchd plist written by `install.py`. The API key is read from `~/.zshrc` by `bin/run-cycle.sh`.

## Working on it
- `uv run pytest` must pass; tests are offline. Add tests for any change to strategy, signals, overlay, btc, paper or scan.
- `signals.track_record` must match `backtest.run` for the same rule (a test enforces this).
- Re-run `uv run tfsa-bot research` after any strategy or data change and compare against the numbers above.
- Don't write to the real `data/tfsa.db` in experiments; monkeypatch `db.DB_PATH` as the tests do.

## Possible next steps
- Review the forward test after ~12 weeks: core vs overlay vs XEQT-like.
- Backtest a wider CAD-listed ETF universe as a portfolio (the watchlist ETFs with proxies for their early history) before adding any of them to `SLEEVES`.
- Optional: hosted (remote) MCP so it works from the phone; phone push via `ntfy_topic`.
- USD account (free on Wealthsimple Premium, which needs $100k; ~$10/month on Core): even with no FX fee the US sleeves gave 14.0% CAGR / -36% DD vs 13.6% / -29% without them, so it is not worth paying for.
- Only if justified: IBKR API for auto-execution (Wealthsimple has no official API).
