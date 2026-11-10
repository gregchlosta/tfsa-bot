"""One cycle of the system. The scheduler runs this ~3x per trading day; the MCP server reuses it.

1. Load the market (only completed daily closes are used for signals).
2. Stress gauge (data) + headlines -> Claude news assessment (if enabled).
3. Once per week (after Friday's close): record the weekly decision for the paper tracks
   (core = backtested strategy, overlay = core + news check, btc = Bitcoin signal). Mid-week,
   news can only cut overlay positions.
4. Report: the strategy's model portfolio (follows config 'follow') and what changed since last
   week, the trend signal of every ETF, the Bitcoin signal, stress gauge and news.
   Analysis and signals only: the bot knows nothing about your money or how you split it.
   The LLM reading the report (Claude app via MCP) does that part with you.
5. Notify when a signal changes.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from . import backtest as bt
from . import btc, data, db, news, notify, overlay, paper, signals
from . import stress as stress_mod
from .report import change_text
from .strategy import Params, scores, target_weights, universe_keys
from .universe import BY_KEY, SLEEVES, Sleeve

NEWS_MAX_AGE_H = 72


def completed_returns(rets: pd.DataFrame) -> pd.DataFrame:
    """Drop today's row while the TSX session is still open (it's a partial bar)."""
    now = datetime.now(data.TZ)
    if rets.index[-1].date() == now.date() and (now.hour, now.minute) < data.CLOSE_FINAL:
        return rets.iloc[:-1]
    return rets


def last_week_end(idx: pd.DatetimeIndex) -> pd.Timestamp:
    """Last trading day of the most recent COMPLETED week."""
    rebs = bt.rebalance_dates(idx, "weekly")
    d = rebs[-1]
    friday = (d + timedelta(days=4 - d.weekday())).date()
    if d.weekday() == 4 or datetime.now(data.TZ).date() > friday:
        return d
    return rebs[-2]


@dataclass(frozen=True)
class Market:
    """Prices as the signals see them: completed closes, and the last completed week."""
    raw: pd.DataFrame   # adjusted closes per ticker (last row may be intraday)
    rets: pd.DataFrame  # daily CAD returns per sleeve, completed closes only
    tri: pd.DataFrame   # total-return index of `rets`
    week: pd.Timestamp  # last trading day of the last completed week

    @property
    def as_of(self) -> pd.Timestamp:
        return self.rets.index[-1]

    @property
    def tri_w(self) -> pd.DataFrame:
        return self.tri.loc[:self.week]

    def close(self, ticker: str) -> float | None:
        """Price at the last completed close (what flip prices are measured from)."""
        return _last(self.raw[ticker].loc[:self.as_of]) if ticker in self.raw else None


def load_market(refresh: bool = False) -> Market:
    rets = completed_returns(data.build_returns(refresh))
    return Market(data.load_raw(), rets, data.total_return_index(rets), last_week_end(rets.index))


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
    m = load_market(refresh)
    week_s = m.week.date().isoformat()

    core_pw = paper.current_weights(m.rets, "core")
    ov_pw = paper.current_weights(m.rets, "overlay")
    stress = stress_mod.snapshot(m.raw, m.rets.index)
    ranking = scores(m.tri_w[universe_keys(p) + ["cash"]], p).sort_values(ascending=False)
    core_now = target_weights(m.tri_w, p, core_pw)

    # --- news ---------------------------------------------------------------
    llm, err, heads, prev_llm = None, None, [], None
    if with_news:
        relevant = sorted((set(_w(core_pw)) | set(_w(ov_pw)) | set(_w(core_now)) | set(ranking.index[:6])) - {"cash"})
        heads = news.headlines(relevant)
        want_llm = cfg["use_llm"] if use_llm is None else use_llm
        if want_llm and news.ensure_api_key():
            ctx = {
                "held": {BY_KEY[k].ticker: v for k, v in _w(ov_pw).items()},
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
        prev_llm = _recent_llm()
        db.save_assessment(m.as_of.date().isoformat(), stress, llm, heads, llm and llm.get("_model"))
    view = llm or _recent_llm()
    risk = view["market_risk"] if view else "normal"
    flags = {f["sleeve"]: f["action"] for f in view["flags"]} if view else {}

    new_week, btc_new_week = _record_paper_tracks(m, p, core_pw, ov_pw, risk, flags)

    # --- the model portfolio: latest decision of the followed track ----------
    decs = db.decisions(cfg["follow"])
    rec = pd.Series(decs[-1]["weights"], dtype=float) if decs else core_now
    prev = pd.Series(decs[-2]["weights"], dtype=float) if len(decs) > 1 else None
    _, shadow_notes = overlay.apply(rec, core_pw, risk, flags)

    summary = {
        "generated": db.now(), "data_as_of": m.as_of.date().isoformat(), "signal_week": week_s,
        "new_week": new_week, "follow": cfg["follow"],
        "stress": stress,
        "news": {k: view[k] for k in ("market_risk", "summary", "key_events", "flags", "confidence")} if view else None,
        "news_fresh": llm is not None, "news_error": err, "headline_count": len(heads),
        "model_portfolio": {BY_KEY[k].ticker: v for k, v in _w(rec).items()},
        "trends": market_trends(m, ranking, rec, p),
        "target_notes": decs[-1]["notes"] if decs else [],
        "news_check_would_change": shadow_notes if cfg["follow"] == "core" else [],
        "changes": None if prev is None else signals.signal_changes(prev, rec),
        "changes_since": decs[-2]["as_of"] if len(decs) > 1 else None,
        "bitcoin": bitcoin_report(m, p, btc_new_week, float(rec.get("btc", 0.0))),
    }
    prev_run = db.last_run() or {}
    db.save_run(summary)
    if send:
        _maybe_notify(summary, prev_llm, prev_run)
    return summary


def _record_paper_tracks(m: Market, p: Params, core_pw: pd.Series | None, ov_pw: pd.Series | None,
                         risk: str, flags: dict[str, str]) -> tuple[bool, bool]:
    """Forward test. Returns (core/overlay decision recorded now, Bitcoin decision recorded now)."""
    week_s = m.week.date().isoformat()
    new_week = not any(d["as_of"] == week_s for d in db.decisions("core"))
    if new_week:
        db.save_decision("core", week_s, _w(target_weights(m.tri_w, p, core_pw)))
        ov_t, notes = overlay.apply(target_weights(m.tri_w, p, ov_pw), ov_pw, risk, flags)
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
            db.save_decision("overlay", m.as_of.date().isoformat(), _w(w), [f"{k}: mid-week cut (news)" for k in cut])

    btc_new_week = not any(d["as_of"] == week_s for d in db.decisions("btc"))
    if btc_new_week:
        db.save_decision("btc", week_s, {"btc": 1.0} if btc.is_in(m.tri_w, p) else {"cash": 1.0})
    return new_week, btc_new_week


def bitcoin_report(m: Market, p: Params, new_week: bool = False, in_model: float = 0.0) -> dict:
    """The recorded weekly Bitcoin signal plus everything btc.status knows."""
    s = BY_KEY["btc"]
    out = btc.status(m.tri_w, m.tri, p, m.close(s.ticker), m.close(s.proxy))
    decs = db.decisions("btc")
    on = bool(decs[-1]["weights"].get("btc")) if decs else out["weekly"]["state"] == "UP"
    out.update({
        "signal": "IN" if on else "OUT",
        "previous_week": ("IN" if decs[-2]["weights"].get("btc") else "OUT") if len(decs) > 1 else None,
        "new_week": new_week,
        "in_model_portfolio": round(in_model, 4),
    })
    return out


def market_trends(m: Market, ranking: pd.Series, model: pd.Series, p: Params) -> list[dict]:
    """Every ETF of the strategy's universe, best trend first (score as of the weekly signal)."""
    rows = []
    for rank, k in enumerate(ranking.index, 1):
        a = signals.asset_signal(m.tri_w, m.tri, k, p)
        w = a["weekly"]
        rows.append({
            "rank": rank, "ticker": BY_KEY[k].ticker, "name": BY_KEY[k].name, "kind": BY_KEY[k].kind,
            "score": w["score"], "trend": w["state"], "since": w["since"], "weeks": w["weeks"],
            "returns": w["returns"], "vs_200d_avg": a["vs_200d_avg"],
            "flip_move": a["latest"]["flip_move"],
            "indicators_up": a["indicators_up"], "indicators_known": a["indicators_known"],
            "in_model": float(model.get(k, 0.0)) > 1e-4,
        })
    return rows


def find_sleeve(query: str) -> Sleeve | None:
    """Match a universe asset by ticker (with or without '.TO') or key, case-insensitive."""
    q = query.strip().upper()
    for s in SLEEVES:
        if q in (s.ticker.upper(), s.ticker.upper().removesuffix(".TO"), s.key.upper()):
            return s
    return None


def asset_report(m: Market, query: str, p: Params = Params()) -> dict:
    """Trend signal for one asset: anything in the universe, or any Yahoo Finance ticker."""
    s = find_sleeve(query)
    if s:
        out = {"ticker": s.ticker, "name": s.name, **signals.asset_signal(m.tri_w, m.tri, s.key, p, m.close(s.ticker))}
        if s.market not in p.markets:
            out["note"] = ("US-listed, measured in CAD. Not in the model portfolio: the ~1.5% FX fee each way "
                           "erased the edge in the backtest.")
        return out
    px = data.load_ticker(query.strip().upper())
    cal = m.rets.index
    a = px.reindex(px.index.union(cal)).ffill(limit=5).reindex(cal)  # same alignment as the universe
    tri = pd.DataFrame({"asset": a / a.dropna().iloc[0], "cash": m.tri["cash"]})
    return {
        "ticker": query.strip().upper(), "name": "(outside the universe)",
        **signals.asset_signal(tri.loc[:m.week], tri, "asset", p, _last(a)),
        "note": ("Outside the backtested universe: same trend maths, in the ticker's own currency against CAD "
                 "cash. Information only, the strategy was never tested on it."),
    }


def _last(a: pd.Series) -> float | None:
    a = a.dropna()
    return float(a.iloc[-1]) if len(a) else None


def _maybe_notify(s: dict, prev_llm: dict | None, prev_run: dict | None = None) -> None:
    msgs = []
    now = datetime.now(data.TZ)
    monday_morning = now.weekday() == 0 and now.hour < 11
    if s["changes"] and (s["new_week"] or monday_morning):
        msgs.append(("New weekly signal", "; ".join(change_text(c) for c in s["changes"][:5])))
    n = s["news"]
    if s["news_fresh"] and n:
        was = (prev_llm or {}).get("market_risk", "normal")
        old_flags = {(f["sleeve"], f["action"]) for f in (prev_llm or {}).get("flags", [])}
        new_flags = [f for f in n["flags"] if (f["sleeve"], f["action"]) not in old_flags]
        if n["market_risk"] != was and n["market_risk"] != "normal":
            msgs.append((f"Market risk: {n['market_risk'].upper()}", n["summary"][:200]))
        for f in new_flags:
            msgs.append((f"News flag: {BY_KEY[f['sleeve']].ticker} {f['action']}", f["reason"][:200]))
    b = s["bitcoin"]
    live = {"UP": "IN", "DOWN": "OUT"}.get(b["latest"]["state"])
    prev_live = (((prev_run or {}).get("bitcoin") or {}).get("latest") or {}).get("state")
    if b["new_week"]:
        if b["previous_week"] and b["previous_week"] != b["signal"]:
            msgs.append((f"Bitcoin trend: {b['signal']}", f"Weekly Bitcoin signal flipped {b['previous_week']} -> {b['signal']}."))
    elif live and live != b["signal"] and b["latest"]["state"] != prev_live:
        msgs.append(("Bitcoin heads-up", f"If the week closed now, the Bitcoin signal would be {live}. "
                     "Only Friday's close confirms it."))
    for title, body in msgs:
        notify.send(title, body)
