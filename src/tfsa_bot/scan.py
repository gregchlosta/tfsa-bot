"""The scan: the same weekly trend signal for everything worth watching on Wealthsimple.

Two tiers, one table:
  model  the 13 ETFs the strategy trades. Only these were backtested as a portfolio.
  watch  `universe.WATCHLIST` plus tickers added in data/config.json: more CAD-listed ETFs, large
         Canadian stocks and CDRs of US stocks. Same maths, information only.

Nothing here reaches `strategy.target_weights`: the scan never changes the model portfolio. Each row
carries what is new about its signal (`signals.flags`: BUY/SELL when it turned at the last week-end,
pending when the latest close disagrees, near a flip) and the track record of following that signal
on that asset (`signals.track_record`), because on most assets the signal alone cut the drawdown but
did not beat holding (`research` section 9).
"""

import pandas as pd

from . import data, signals
from .strategy import Params, universe_keys
from .universe import BY_KEY, WATCHLIST, Watch

KINDS = ("equity", "sector", "real_estate", "bond", "commodity", "crypto", "stock", "cdr", "custom")
NOTE = ("tier 'model' = the 13 ETFs the backtested strategy trades; tier 'watch' = same trend maths, never "
        "backtested as a portfolio. 'record' compares following the signal on that asset alone with holding "
        "it: on most assets the signal lowered the drawdown but not the return. CDRs are measured on the US "
        "stock in USD (what a CAD-hedged CDR tracks) less 0.6%/yr; their flip price is the US stock's price.")


def _bare(ticker: str) -> str:
    return ticker.upper().removesuffix(".TO").removesuffix(".NE")


def watchlist(cfg: dict, p: Params = Params()) -> list[Watch]:
    """Default watchlist minus `watchlist_hidden`, plus `watchlist_extra` (any Yahoo ticker)."""
    hidden = {t.upper() for t in cfg.get("watchlist_hidden") or []}
    items = [w for w in WATCHLIST if w.ticker.upper() not in hidden]
    have = {w.ticker.upper() for w in items} | {BY_KEY[k].ticker.upper() for k in universe_keys(p)}
    for t in cfg.get("watchlist_extra") or []:
        t = t.strip().upper()
        if t and t not in have:
            have.add(t)
            items.append(Watch(t, "(added by you)", "custom"))
    return items


def edit_watchlist(cfg: dict, add: list[str] | None = None, remove: list[str] | None = None) -> dict:
    """Add or remove scan tickers in the config (returned, not saved). Removing a default entry hides it;
    adding it again un-hides it."""
    default = {w.ticker.upper() for w in WATCHLIST}
    extra = [t.upper() for t in cfg.get("watchlist_extra") or []]
    hidden = [t.upper() for t in cfg.get("watchlist_hidden") or []]
    for t in (x.strip().upper() for x in add or []):
        if t in hidden:
            hidden.remove(t)
        elif t and t not in default and t not in extra:
            extra.append(t)
    for t in (x.strip().upper() for x in remove or []):
        if t in extra:
            extra.remove(t)
        elif t in default and t not in hidden:
            hidden.append(t)
    return {**cfg, "watchlist_extra": extra, "watchlist_hidden": hidden}


def find_watch(query: str, cfg: dict) -> Watch | None:
    """Match a watchlist entry by ticker (with or without .TO/.NE) or by the ticker its history comes from."""
    q = query.strip().upper()
    for w in watchlist(cfg):
        if q in (w.ticker.upper(), _bare(w.ticker), w.history.upper()):
            return w
    return None


def watch_signal(m, w: Watch, px: pd.Series, p: Params = Params()) -> tuple[dict, dict | None] | None:
    """(asset signal, track record) for one watchlist entry from its price history; None without prices."""
    lvl = data.series_index(px.dropna(), m.rets.index, w.drag) if px.notna().any() else None
    if lvl is None or lvl.dropna().empty:
        return None
    tri = pd.DataFrame({"asset": lvl, "cash": m.tri["cash"]})
    done = px.dropna().loc[:m.as_of]
    close = float(done.iloc[-1]) if len(done) else None
    return signals.asset_signal(tri.loc[:m.week], tri, "asset", p, close), signals.track_record(tri, "asset", p)


def _row(ticker: str, name: str, kind: str, tier: str, a: dict | None, rec: dict | None, **extra) -> dict:
    if a is None:
        return {"ticker": ticker, "name": name, "kind": kind, "tier": tier, "trend": None, "score": None,
                "weeks": 0, "event": None, "pending": None, "near_flip": False, "record": None, **extra}
    w, n = a["weekly"], a["latest"]
    fast_up = any(i["name"] == "fast_score" and i["state"] == "UP" for i in a.get("indicators") or [])
    return {
        "ticker": ticker, "name": name, "kind": kind, "tier": tier,
        "trend": w["state"], "score": w["score"], "since": w["since"], "weeks": w["weeks"],
        "event": a["event"], "pending": a["pending"], "near_flip": a["near_flip"],
        "flip_move": n["flip_move"], "flip_price": n.get("flip_price"), "close": n.get("close"),
        "weekly_move": a["weekly_move"], "flip_weeks": a.get("flip_weeks"),
        "ann_vol": a.get("ann_vol"), "risk_adj_score": a.get("risk_adj_score"), "rsi_14": a.get("rsi_14"),
        "fast_up": fast_up,
        "returns": w["returns"], "vs_200d_avg": a["vs_200d_avg"], "from_3y_high": a.get("from_3y_high"),
        "indicators_up": a["indicators_up"], "indicators_known": a["indicators_known"],
        "record": rec, **extra,
    }


def build(m, cfg: dict, p: Params = Params(), refresh: bool = False, held: set[str] | None = None) -> dict:
    """Every scanned asset, best trend first, with this week's events. `held` = sleeve keys in the model
    portfolio."""
    held = held or set()
    cal = m.rets.index
    rows = []

    keys = universe_keys(p)
    old = data.stale(m.raw.reindex(columns=[BY_KEY[k].ticker for k in keys]), cal, m.as_of)
    for k in keys:
        s = BY_KEY[k]
        a = signals.asset_signal(m.tri_w, m.tri, k, p, m.close(s.ticker))
        rows.append(_row(s.ticker, s.name, s.kind, "model", a, signals.track_record(m.tri, k, p),
                         in_model=k in held, stale=old.get(s.ticker)))

    items = watchlist(cfg, p)
    px = data.load_watch([w.history for w in items], refresh)
    old = data.stale(px, cal, m.as_of)
    for w in items:
        got = watch_signal(m, w, px[w.history], p)
        a, rec = got if got else (None, None)
        extra = {"in_model": False, "stale": old.get(w.history)}
        if w.source:
            extra["priced_as"] = w.source
        rows.append(_row(w.ticker, w.name, w.kind, "watch", a, rec, **extra))

    rows.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0.0)))
    return {
        "data_as_of": m.as_of.date().isoformat(), "signal_week": m.week.date().isoformat(),
        "rows": rows, "events": events(rows),
        "counts": {"scanned": len(rows), "up": sum(r["trend"] == "UP" for r in rows),
                   "down": sum(r["trend"] == "DOWN" for r in rows),
                   "no_signal": sum(r["trend"] is None for r in rows)},
        "stale": {r["ticker"]: r["stale"] for r in rows if r["stale"]},
        "note": NOTE,
    }


def events(rows: list[dict]) -> dict:
    """What changed or is about to: confirmed flips at the last week-end (buy / sell), latest closes
    that disagree with the weekly signal (pending), and signals within one typical week of flipping.
    Assets with stale prices are left out."""
    def brief(r: dict) -> dict:
        return {k: r.get(k) for k in ("ticker", "name", "kind", "tier", "trend", "score", "flip_move")}

    live = [r for r in rows if not r["stale"]]
    return {
        "buy": [brief(r) for r in live if r["event"] == "BUY"],
        "sell": [brief(r) for r in live if r["event"] == "SELL"],
        "pending_up": [brief(r) for r in live if r["pending"] == "UP"],
        "pending_down": [brief(r) for r in live if r["pending"] == "DOWN"],
        "near_flip": [brief(r) for r in live if r["near_flip"]],
    }


def select(
    rows: list[dict],
    kind: str | None = None,
    show: str = "all",
    tier: str | None = None,
    sort: str = "score",
) -> list[dict]:
    """Filter and sort scan rows.
    show: all | up | down | events (flip, pending flip, or near one).
    tier: model | watch | None (both).
    sort: score | risk_adj | flip (closest to flipping first)."""
    out = [r for r in rows if (kind is None or r["kind"] == kind) and (tier is None or r["tier"] == tier)]
    if show == "up":
        out = [r for r in out if r["trend"] == "UP"]
    elif show == "down":
        out = [r for r in out if r["trend"] == "DOWN"]
    elif show == "events":
        out = [r for r in out if r["event"] or r["pending"] or r["near_flip"]]
    if sort == "risk_adj":
        out = sorted(out, key=lambda r: (r.get("risk_adj_score") is None, -(r.get("risk_adj_score") or 0.0)))
    elif sort == "flip":
        out = sorted(out, key=lambda r: (r.get("flip_weeks") is None, r.get("flip_weeks") or 999.0))
    return out


def opportunities(sc: dict, kind: str | None = None, tier: str | None = None, limit: int = 8) -> dict:
    """Distill the scan into actionable buy/sell setups and risk warnings (context only for watchlist)."""
    live = [r for r in sc["rows"] if not r.get("stale") and r.get("trend") is not None
            and (kind is None or r["kind"] == kind) and (tier is None or r["tier"] == tier)]

    def item(r: dict) -> dict:
        rec = r.get("record")
        rec_beats = bool(rec and rec["signal_cagr"] >= rec["hold_cagr"] - 0.005)
        rec_dd_cut = round(rec["signal_max_dd"] - rec["hold_max_dd"], 3) if rec else None
        return {
            "ticker": r["ticker"], "name": r["name"], "kind": r["kind"], "tier": r["tier"],
            "in_model": r.get("in_model", False), "trend": r["trend"], "weeks": r["weeks"],
            "score": r["score"], "risk_adj_score": r.get("risk_adj_score"),
            "rsi_14": r.get("rsi_14"), "vs_200d_avg": r.get("vs_200d_avg"),
            "flip_move": r.get("flip_move"), "flip_price": r.get("flip_price"),
            "flip_weeks": r.get("flip_weeks"), "close": r.get("close"),
            "indicators": f"{r.get('indicators_up', 0)}/{r.get('indicators_known', 0)}",
            "record_beats_hold": rec_beats if rec else None,
            "record_dd_saved": rec_dd_cut,
            "record": rec,
        }

    by_quality = lambda r: -(r.get("risk_adj_score") or -999.0)
    by_score = lambda r: -(r.get("score") or -999.0)
    by_flip = lambda r: r.get("flip_weeks") if r.get("flip_weeks") is not None else 999.0

    fresh_buys = [item(r) for r in sorted((r for r in live if r["event"] == "BUY"), key=by_quality)[:limit]]
    fresh_sells = [item(r) for r in sorted((r for r in live if r["event"] == "SELL"), key=by_score)[:limit]]
    pending_up = [item(r) for r in sorted((r for r in live if r["pending"] == "UP"), key=by_quality)[:limit]]
    pending_down = [item(r) for r in sorted((r for r in live if r["pending"] == "DOWN"), key=by_flip)[:limit]]

    # Clean, high-quality uptrends (not overextended)
    strongest_up = [
        item(r) for r in sorted(
            (r for r in live if r["trend"] == "UP" and (r.get("indicators_up") or 0) >= 4),
            key=by_quality,
        )[:limit]
    ]

    # Pullback-in-uptrend: confirmed UP, 4+/5 indicators up, not pending down, RSI cooled off <= 48
    pullbacks = [
        item(r) for r in sorted(
            (r for r in live if r["trend"] == "UP" and not r["pending"]
             and (r.get("indicators_up") or 0) >= 4
             and r.get("rsi_14") is not None and r["rsi_14"] <= 48.0),
            key=by_quality,
        )[:limit]
    ]

    # Early turnaround radar: currently DOWN, but fast 1/3m score already UP and within 2 weekly moves of flipping
    early_turns = [
        item(r) for r in sorted(
            (r for r in live if r["trend"] == "DOWN" and r.get("fast_up")
             and r.get("flip_weeks") is not None and r["flip_weeks"] <= 2.5),
            key=by_flip,
        )[:limit]
    ]

    # Extended / stretched uptrends: RSI >= 75 or >35% above 200d avg (chasing risk)
    extended = [
        item(r) for r in sorted(
            (r for r in live if r["trend"] == "UP" and (
                (r.get("rsi_14") is not None and r["rsi_14"] >= 75.0)
                or (r.get("vs_200d_avg") is not None and r["vs_200d_avg"] >= 0.35)
            )),
            key=lambda r: -(r.get("rsi_14") or 0.0),
        )[:limit]
    ]

    return {
        "data_as_of": sc["data_as_of"],
        "signal_week": sc["signal_week"],
        "counts": sc["counts"],
        "fresh_buys": fresh_buys,
        "fresh_sells": fresh_sells,
        "pending_up": pending_up,
        "pending_down": pending_down,
        "strongest_uptrends": strongest_up,
        "pullback_setups": pullbacks,
        "early_turnarounds": early_turns,
        "extended_warnings": extended,
        "note": (
            "Confirmed weekly signals only change at Friday's close. Pullbacks, early turnarounds, RSI and "
            "risk-adjusted scores are timing/quality context, not separate signals. Only the 13 [model] ETFs "
            "were backtested as a portfolio."
        ),
    }
