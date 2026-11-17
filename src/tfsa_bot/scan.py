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
    return {
        "ticker": ticker, "name": name, "kind": kind, "tier": tier,
        "trend": w["state"], "score": w["score"], "since": w["since"], "weeks": w["weeks"],
        "event": a["event"], "pending": a["pending"], "near_flip": a["near_flip"],
        "flip_move": n["flip_move"], "flip_price": n.get("flip_price"), "weekly_move": a["weekly_move"],
        "returns": w["returns"], "vs_200d_avg": a["vs_200d_avg"],
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


def select(rows: list[dict], kind: str | None = None, show: str = "all") -> list[dict]:
    """Filter scan rows. show: all | up | down | events (anything with a flip, a pending flip or near one)."""
    out = [r for r in rows if kind is None or r["kind"] == kind]
    if show == "up":
        out = [r for r in out if r["trend"] == "UP"]
    elif show == "down":
        out = [r for r in out if r["trend"] == "DOWN"]
    elif show == "events":
        out = [r for r in out if r["event"] or r["pending"] or r["near_flip"]]
    return out
