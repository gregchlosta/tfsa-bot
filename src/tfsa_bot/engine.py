"""One cycle of the system. The scheduler runs this ~3x per trading day; the MCP server reuses it.

1. Refresh prices (only completed daily closes are used for signals).
2. Stress gauge (data) + headlines -> Claude news assessment (if enabled).
3. Once per week (after Friday's close): record the weekly decision for both paper tracks
   (core = backtested strategy, overlay = core + news check). Mid-week, news can only cut
   overlay positions.
4. Build YOUR recommendation (follows config 'follow') and the exact trades vs your account.
5. Notify when there's something to do.
"""

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from . import backtest as bt
from . import db, news, notify, overlay, paper, portfolio
from . import stress as stress_mod
from .data import build_returns, load_raw, total_return_index
from .strategy import Params, scores, target_weights, universe_keys
from .universe import BY_KEY

TZ = ZoneInfo("America/Toronto")
NEWS_MAX_AGE_H = 72


def completed_returns(rets: pd.DataFrame) -> pd.DataFrame:
    """Drop today's row while the TSX session is still open (it's a partial bar)."""
    now = datetime.now(TZ)
    if rets.index[-1].date() == now.date() and (now.hour, now.minute) < (16, 30):
        return rets.iloc[:-1]
    return rets


def last_week_end(idx: pd.DatetimeIndex) -> pd.Timestamp:
    """Last trading day of the most recent COMPLETED week."""
    rebs = bt.rebalance_dates(idx, "weekly")
    d = rebs[-1]
    friday = (d + timedelta(days=4 - d.weekday())).date()
    if d.weekday() == 4 or datetime.now(TZ).date() > friday:
        return d
    return rebs[-2]


def _recent_llm() -> dict | None:
    a = db.latest_assessment(with_llm=True)
    if not a:
        return None
    age = datetime.now() - datetime.fromisoformat(a["created_at"])
    return a["llm"] if age < timedelta(hours=NEWS_MAX_AGE_H) else None


def _w(s: pd.Series | None) -> dict:
    return {} if s is None else {k: round(float(v), 4) for k, v in s.items() if v > 1e-4}


def run_cycle(refresh: bool = True, with_news: bool = True, use_llm: bool | None = None, send: bool = True) -> dict:
    cfg = db.config()
    p = Params()
    rets = completed_returns(build_returns(refresh))
    raw = load_raw()
    prices = raw.ffill().iloc[-1]  # freshest prices (may be intraday) for sizing orders
    tri = total_return_index(rets)
    as_of = rets.index[-1]
    week = last_week_end(rets.index)
    tri_w = tri.loc[:week]
    week_s = week.date().isoformat()

    acct = portfolio.state(prices)
    acct_w = portfolio.current_weights(acct)
    core_pw = paper.current_weights(rets, "core")
    ov_pw = paper.current_weights(rets, "overlay")
    stress = stress_mod.snapshot(raw, rets.index)
    ranking = scores(tri_w[universe_keys(p) + ["cash"]], p).sort_values(ascending=False)
    core_now = target_weights(tri_w, p, acct_w if acct_w is not None else core_pw)

    # --- news ---------------------------------------------------------------
    llm, err, heads = None, None, []
    held_keys = set(_w(acct_w)) | set(_w(ov_pw))
    relevant = sorted((held_keys | set(_w(core_now)) | set(ranking.index[:6])) - {"cash"})
    if with_news:
        heads = news.headlines(relevant)
        want_llm = cfg["use_llm"] if use_llm is None else use_llm
        if want_llm and os.environ.get("ANTHROPIC_API_KEY"):
            ctx = {
                "held": {BY_KEY[k].ticker: v for k, v in _w(acct_w if acct_w is not None else ov_pw).items()},
                "targets": {BY_KEY[k].ticker: v for k, v in _w(core_now).items()},
                "ranking": {BY_KEY[k].ticker: round(float(v), 3) for k, v in ranking.items()},
                "relevant_sleeves": relevant,
            }
            try:
                llm = news.assess(heads, stress, ctx)
            except Exception as e:  # never let the news layer break the core
                err = f"{type(e).__name__}: {e}"
        elif want_llm:
            err = "ANTHROPIC_API_KEY not set - news assessment skipped"
        prev = _recent_llm()
        db.save_assessment(as_of.date().isoformat(), stress, llm, heads, llm and llm.get("_model"))
    else:
        prev = None
    view = llm or _recent_llm()
    risk = view["market_risk"] if view else "normal"
    flags = {f["sleeve"]: f["action"] for f in view["flags"]} if view else {}

    # --- paper tracks (forward test) ----------------------------------------
    new_week = not any(d["as_of"] == week_s for d in db.decisions("core"))
    if new_week:
        db.save_decision("core", week_s, _w(target_weights(tri_w, p, core_pw)))
        ov_t, notes = overlay.apply(target_weights(tri_w, p, ov_pw), ov_pw, risk, flags)
        db.save_decision("overlay", week_s, _w(ov_t), notes)
    elif ov_pw is not None:
        # Mid-week, news can only cut. "reduce" = at most half the week's CORE weight; measuring
        # against the core decision (never overwritten) keeps repeated runs from compounding cuts.
        core_week = {d["as_of"]: d["weights"] for d in db.decisions("core")}.get(week_s, {})
        w = ov_pw.copy()
        cut = [k for k, a in flags.items() if a == "reduce" and w.get(k, 0) > core_week.get(k, 0) / 2 + 0.01]
        if cut:
            for k in cut:
                w[k] = core_week.get(k, 0) / 2
            w["cash"] = 1 - w.drop("cash").sum()
            db.save_decision("overlay", as_of.date().isoformat(), _w(w), [f"{k}: mid-week cut (news)" for k in cut])

    # --- your recommendation -----------------------------------------------
    rec, rec_notes = core_now, []
    if cfg["follow"] == "overlay":
        rec, rec_notes = overlay.apply(core_now, acct_w, risk, flags)
    elif acct_w is None:
        last = db.decisions("core")
        rec = pd.Series(last[-1]["weights"]) if last else core_now
    shadow, shadow_notes = overlay.apply(rec, acct_w, risk, flags)
    orders = portfolio.orders(rec, acct, prices, cfg["default_capital"], cfg["min_trade_dollars"])

    summary = {
        "generated": db.now(), "data_as_of": as_of.date().isoformat(), "signal_week": week_s,
        "new_week": new_week, "follow": cfg["follow"],
        "stress": stress,
        "news": {k: view[k] for k in ("market_risk", "summary", "key_events", "flags", "confidence")} if view else None,
        "news_fresh": llm is not None, "news_error": err, "headline_count": len(heads),
        "target": {BY_KEY[k].ticker: v for k, v in _w(rec).items()},
        "target_notes": rec_notes,
        "news_check_would_change": shadow_notes if cfg["follow"] == "core" else [],
        "orders": orders,
        "account": acct,
    }
    db.save_run(summary)
    if send:
        _maybe_notify(summary, prev)
    return summary


def _maybe_notify(s: dict, prev_llm: dict | None) -> None:
    msgs = []
    now = datetime.now(TZ)
    monday_morning = now.weekday() == 0 and now.hour < 11
    if s["orders"] and (s["new_week"] or monday_morning):
        sides = ", ".join(f"{o['side']} {o['ticker']} ${o['dollars']:,.0f}" for o in s["orders"][:4])
        msgs.append(("Rebalance ready", f"{len(s['orders'])} trades: {sides}"))
    n = s["news"]
    if s["news_fresh"] and n:
        was = (prev_llm or {}).get("market_risk", "normal")
        old_flags = {(f["sleeve"], f["action"]) for f in (prev_llm or {}).get("flags", [])}
        new_flags = [f for f in n["flags"] if (f["sleeve"], f["action"]) not in old_flags]
        if n["market_risk"] != was and n["market_risk"] != "normal":
            msgs.append((f"Market risk: {n['market_risk'].upper()}", n["summary"][:200]))
        for f in new_flags:
            msgs.append((f"News flag: {BY_KEY[f['sleeve']].ticker} {f['action']}", f["reason"][:200]))
    for title, body in msgs:
        notify.send(title, body)


def brief_text(s: dict) -> str:
    L = [f"TFSA bot — data as of {s['data_as_of']} close (weekly signal from {s['signal_week']}), following: {s['follow']}"]
    a = s["account"]
    if a["has_account"]:
        L.append(f"Account: ${a['total']:,.2f}  (deposits ${a['net_deposits']:,.2f}, gain ${a['gain']:+,.2f}, cash ${a['cash']:,.2f})")
    else:
        L.append("Account: nothing logged yet — sizes below assume default capital. Log deposits/trades to track.")
    L.append("\nTARGET: " + ", ".join(f"{t} {w:.0%}" for t, w in s["target"].items()))
    if s["orders"]:
        L.append("TRADES TO PLACE:")
        for o in s["orders"]:
            L.append(f"  {o['side']:4s} {o['ticker']:9s} ≈{o['shares']:.3f} sh  (${o['dollars']:,.0f} @ {o['price']})  {o['name']}")
    else:
        L.append("No trades needed.")
    st = s["stress"]
    L.append(f"\nSTRESS GAUGE: {st['level']} ({', '.join(st['fired']) or 'nothing firing'}; VIX {st['vix']})")
    n = s["news"]
    if n:
        L.append(f"NEWS CHECK ({'fresh' if s['news_fresh'] else 'from earlier'}, confidence {n['confidence']}): risk {n['market_risk']}")
        L.append("  " + n["summary"])
        for f in n["flags"]:
            L.append(f"  FLAG {BY_KEY[f['sleeve']].ticker} {f['action']}: {f['reason']}")
    if s.get("news_error"):
        L.append(f"  (news: {s['news_error']})")
    if s["news_check_would_change"]:
        L.append("  Shadow overlay would: " + "; ".join(s["news_check_would_change"]))
    return "\n".join(L)
