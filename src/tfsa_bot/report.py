"""Plain-text rendering of the report dicts (CLI, notifications, the `text` field of MCP tools)."""

from .universe import BY_KEY

INDICATOR_LABELS = {
    "trend_score": "trend score 1/3/6/12m",
    "fast_score": "fast score 1/3m",
    "above_100d": "price > 100-day avg",
    "above_200d": "price > 200-day avg",
    "ma50_above_ma200": "50-day > 200-day avg",
}
ROLE_NOTES = {
    "signal": "<- the signal (backtested)",
    "early": "early; more false starts",
    "confirm": "about as fast as the signal",
    "slow": "slow confirmation",
}


def pct(v: float | None, digits: int = 1) -> str:
    return "n/a" if v is None else f"{v:+.{digits}%}"


def _flip(move: float | None) -> str:
    return "" if move is None else f"flips on a {abs(move):.0%} {'drop' if move < 0 else 'rise'}"


def _weeks(n: int) -> str:
    return f"{n} week{'' if n == 1 else 's'}"


def change_text(c: dict) -> str:
    if c["action"] == "BUY":
        return f"BUY {c['ticker']} -> {c['to']:.0%}"
    if c["action"] == "SELL":
        return f"SELL all {c['ticker']} (was {c['from']:.0%})"
    return f"{c['action']} {c['ticker']} {c['from']:.0%} -> {c['to']:.0%}"


def brief_text(s: dict) -> str:
    L = [f"Signals — data as of {s['data_as_of']} close (weekly signal from {s['signal_week']}), following: {s['follow']}"]
    L.append("\nSTRATEGY MODEL PORTFOLIO (top 4 uptrends, equal weight): "
             + ", ".join(f"{t} {w:.1%}".replace(".0%", "%") for t, w in s["model_portfolio"].items()))
    for note in s.get("target_notes") or []:
        L.append(f"  note: {note}")
    if s["changes"] is None:
        L.append("First weekly signal: nothing to compare with yet.")
    elif s["changes"]:
        L.append(f"CHANGES since the {s['changes_since']} signal:")
        for c in s["changes"]:
            L.append(f"  {change_text(c):40s} {c['name']}")
    else:
        L.append(f"No changes since the {s['changes_since']} signal.")
    L.append("\nTRENDS (score = avg 1/3/6/12m return minus cash; > 0 = uptrend; weeks = how long the trend has held;"
             "\n        flip = move from the latest close that would turn it; n/5 = trend indicators pointing up):")
    for t in s["trends"]:
        r = t["returns"]
        L.append(f"  {t['rank']:2d}. {t['ticker']:10s} {t['score']:+.3f} {t['trend']:4s} {t['weeks']:3d}w  "
                 f"1m {pct(r['21d'])} 3m {pct(r['63d'])} 6m {pct(r['126d'])} 12m {pct(r['252d'])}  "
                 f"vs200d {pct(t['vs_200d_avg'])}  flip {pct(t['flip_move'], 0)}  "
                 f"{t['indicators_up']}/{t['indicators_known']}"
                 f"{'  [held]' if t['in_model'] else ''}  {t['name']}")
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
    L.append("\n" + btc_text(s["bitcoin"]))
    return "\n".join(L)


def signal_lines(a: dict) -> list[str]:
    """Body shared by every asset: latest close vs the weekly signal, returns, indicator panel."""
    n = a["latest"]
    L = []
    if n["state"]:
        at = f" ({a.get('ticker', 'price')} ≈ ${n['flip_price']}, last close ${n['close']})" if n.get("flip_price") else ""
        usd = f"; Bitcoin ≈ US${n['flip_price_btc_usd']:,.0f}" if n.get("flip_price_btc_usd") else ""
        L.append(f"  Latest close {n['as_of']}: score {n['score']:+.3f} -> {n['state']}. {_flip(n['flip_move']).capitalize()}{at}{usd}.")
    L.append("  Returns 1/3/6/12m: " + ", ".join(pct(v) for v in n["returns"].values())
             + "  (cash: " + ", ".join(pct(v) for v in a["cash_returns"].values()) + ")")
    L.append(f"  vs 200-day average {pct(a['vs_200d_avg'])}, from 3-year high {pct(a['from_3y_high'])}")
    L.append(f"  Indicators at the latest close ({a['indicators_up']} of {a['indicators_known']} up; only the first is the signal):")
    for i in a["indicators"]:
        L.append(f"    {INDICATOR_LABELS[i['name']]:22s} {i['state'] or 'n/a':5s} {_flip(i['flip_move']):20s} {ROLE_NOTES[i['role']]}")
    return L


def signal_text(a: dict) -> str:
    w = a["weekly"]
    if not w["state"]:
        head = f"{a['ticker']} {a['name']} — no signal yet (needs 12 months of history)"
    else:
        head = (f"{a['ticker']} {a['name']} — weekly trend {w['state']} since {w['since']} ({_weeks(w['weeks'])}), "
                f"score {w['score']:+.3f} as of {w['as_of']}")
    L = [head] + signal_lines(a)
    if a.get("note"):
        L.append(f"  Note: {a['note']}")
    return "\n".join(L)


def btc_text(b: dict) -> str:
    w, n = b["weekly"], b["latest"]
    was = f", was {b['previous_week']} last week" if b["previous_week"] and b["previous_week"] != b["signal"] else ""
    L = [f"BITCOIN TREND SIGNAL: {b['signal']} ({'hold FBTC.TO' if b['signal'] == 'IN' else 'out, cash'}{was}) "
         f"— weekly score {w['score']:+.3f}, {w['state']} since {w['since']} ({_weeks(w['weeks'])})"]
    if b["in_model_portfolio"]:
        L.append(f"  The strategy model portfolio also holds FBTC at {b['in_model_portfolio']:.0%} (cap 15%).")
    L += signal_lines({**b, "ticker": "FBTC"})
    if n["state"] and (n["state"] == "UP") != (b["signal"] == "IN"):
        L.append("  HEADS-UP: the latest close disagrees with the weekly signal. Act only if Friday's close confirms.")
    c = b.get("cycle")
    if c:
        L.append("  CYCLE CLOCK (information only: three past cycles are a description, not evidence):")
        L.append(f"    {c['days_since_halving']} days since the {c['last_halving']} halving. Cycle high {c['cycle_high_date']} "
                 f"({c['days_since_cycle_high']} days ago), now {pct(c['from_cycle_high'], 0)} from it. "
                 f"Low since then {c['low_since_high_date']} ({pct(c['low_since_high'], 0)}); now {pct(c['above_that_low'], 0)} above that low.")
        for e in c["same_point_in_earlier_cycles"]:
            nxt = ", ".join(f"{k} {pct(v, 0)}" for k, v in e["next"].items())
            frm = "" if e["from_cycle_high"] is None else f" was {pct(e['from_cycle_high'], 0)} from its cycle high;"
            L.append(f"    Same day-count after the {e['halving'][:4]} halving ({e['date']}):{frm} next {nxt} (trading days, CAD)")
    return "\n".join(L)
