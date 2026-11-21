"""Plain-text rendering of the report dicts (CLI, notifications, the `text` field of MCP tools)."""

from .universe import BY_KEY

INDICATOR_LABELS = {
    "trend_score": "trend score 1/3/6/12m",
    "fast_score": "fast score 1/3m",
    "above_100d": "price > 100-day avg",
    "above_200d": "price > 200-day avg",
    "ma50_above_ma200": "50-day > 200-day avg",
}
MODEL_TAGS = {"held": "  [held]", "held_by_buffer": "  [held: buffer]", "next_in_line": "  [next in line]", None: ""}
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


def record_text(r: dict | None) -> str:
    """Following the signal on this asset alone vs holding it: return per year / max drawdown."""
    if not r:
        return "no track record yet"
    return (f"signal {r['signal_cagr']:+.0%}/{r['signal_max_dd']:.0%} vs hold {r['hold_cagr']:+.0%}/{r['hold_max_dd']:.0%} "
            f"({r['years']:.0f}y)")


def _names(items: list[dict], limit: int = 12, flip: bool = False) -> str:
    more = f" +{len(items) - limit} more" if len(items) > limit else ""
    return ", ".join(i["ticker"] + (f" ({pct(i['flip_move'], 0)})" if flip and i.get("flip_move") is not None else "")
                     for i in items[:limit]) + more


def events_lines(ev: dict, week: str) -> list[str]:
    """This week's signal events: confirmed flips, unconfirmed ones, and signals close to flipping."""
    L = []
    if ev["buy"]:
        L.append(f"  BUY  (trend turned UP at the {week} close):   {_names(ev['buy'])}")
    if ev["sell"]:
        L.append(f"  SELL (trend turned DOWN at the {week} close): {_names(ev['sell'])}")
    if not ev["buy"] and not ev["sell"]:
        L.append(f"  No trend flipped at the {week} close.")
    pend = [f"would turn {d.upper()}: {_names(ev[f'pending_{d}'])}" for d in ("up", "down") if ev[f"pending_{d}"]]
    if pend:
        L.append("  Pending (the latest close disagrees with the weekly signal; only the week's last close confirms): "
                 + "; ".join(pend))
    if ev["near_flip"]:
        L.append("  Near a flip (within one typical week's move; move that flips it): " + _names(ev["near_flip"], flip=True))
    return L


def _stale(v: dict) -> str:
    return "no price data" if not v["last"] else f"last price {v['last']}, {v['trading_days_old']} trading days old"


def brief_text(s: dict) -> str:
    L = [f"Signals — data as of {s['data_as_of']} close (weekly signal from {s['signal_week']}), following: {s['follow']}"]
    if s.get("data_warnings"):
        L.append("DATA WARNING, signals using these prices are stale: "
                 + "; ".join(f"{w['ticker']} ({_stale(w)})" for w in s["data_warnings"]))
    rule = s.get("model_rule") or {"top_n": 4, "hold_buffer": 2}
    top, keep = rule["top_n"], rule["top_n"] + rule["hold_buffer"]
    L.append(f"\nSTRATEGY MODEL PORTFOLIO (top {top} uptrends, equal weight; a holding stays while it ranks in the top {keep}): "
             + ", ".join(f"{t} {w:.1%}".replace(".0%", "%") for t, w in s["model_portfolio"].items()))
    kept = [f"{t['ticker']} (rank {t['rank']})" for t in s["trends"] if t.get("model_status") == "held_by_buffer"]
    waits = [f"{t['ticker']} (rank {t['rank']})" for t in s["trends"] if t.get("model_status") == "next_in_line"]
    if kept:
        L.append(f"  Kept by the hold buffer: {', '.join(kept)}."
                 + (f" Next in line, enters when a holding drops out: {', '.join(waits)}." if waits else ""))
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
                 f"{MODEL_TAGS.get(t.get('model_status') or ('held' if t['in_model'] else None), '')}  {t['name']}")
    sc = s.get("scan") or {}
    if sc.get("events"):
        c = sc["counts"]
        L.append(f"\nSIGNAL EVENTS across the scan ({c['scanned']} assets: {c['up']} UP, {c['down']} DOWN; the watchlist is "
                 "information only; `scan` shows the full table):")
        L += events_lines(sc["events"], s["signal_week"])
        if sc.get("stale"):
            L.append("  Stale prices, left out of the events: " + "; ".join(f"{t} ({_stale(v)})" for t, v in sc["stale"].items()))
    elif sc.get("error"):
        L.append(f"\nSIGNAL EVENTS: scan unavailable ({sc['error']})")
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
        # Flip prices next to the latest price, so a flip level is never read as the current price.
        where = []
        if n.get("flip_price"):
            where.append(f"at {a.get('price_ticker') or a.get('ticker', 'price')} ≈ ${n['flip_price']}, last close ${n['close']}")
        if n.get("flip_price_btc_usd"):
            where.append(f"at Bitcoin ≈ US${n['flip_price_btc_usd']:,.0f}, now US${n['btc_usd']:,.0f}")
        at = f" ({'; '.join(where)})" if where else ""
        L.append(f"  Latest close {n['as_of']}: score {n['score']:+.3f} -> {n['state']}. {_flip(n['flip_move']).capitalize()}{at}.")
    L.append("  Returns 1/3/6/12m: " + ", ".join(pct(v) for v in n["returns"].values())
             + "  (cash: " + ", ".join(pct(v) for v in a["cash_returns"].values()) + ")")
    L.append(f"  vs 200-day average {pct(a['vs_200d_avg'])}, from 3-year high {pct(a['from_3y_high'])}")
    extras = []
    if a.get("risk_adj_score") is not None:
        extras.append(f"risk-adj score {a['risk_adj_score']:+.2f}")
    if a.get("ann_vol") is not None:
        extras.append(f"vol {a['ann_vol']:.0%}")
    if a.get("rsi_14") is not None:
        tag = " (stretched)" if a["rsi_14"] >= 75 else " (oversold)" if a["rsi_14"] <= 35 else ""
        extras.append(f"RSI(14) {a['rsi_14']:.0f}{tag}")
    if a.get("flip_weeks") is not None:
        extras.append(f"~{a['flip_weeks']:.1f} typical wks to flip")
    if extras:
        L.append("  Quality & timing (context): " + ", ".join(extras))
    L.append(f"  Indicators at the latest close ({a['indicators_up']} of {a['indicators_known']} up; only the first is the signal):")
    for i in a["indicators"]:
        L.append(f"    {INDICATOR_LABELS[i['name']]:22s} {i['state'] or 'n/a':5s} {_flip(i['flip_move']):20s} {ROLE_NOTES[i['role']]}")
    r = a.get("record")
    if r:
        L.append(f"  Track record of following this signal on its own since {r['from'][:4]} ({r['years']:.0f} years): "
                 f"{r['signal_cagr']:+.1%}/yr, max drawdown {r['signal_max_dd']:.0%}, {r['flips_per_year']} flips/yr, "
                 f"in the market {r['time_in']:.0%} of the time. Holding it throughout: {r['hold_cagr']:+.1%}/yr, "
                 f"max drawdown {r['hold_max_dd']:.0%}.")
    return L


def signal_text(a: dict) -> str:
    w = a["weekly"]
    if not w["state"]:
        head = f"{a['ticker']} {a['name']} — no signal yet (needs 12 months of history)"
    else:
        head = (f"{a['ticker']} {a['name']} — weekly trend {w['state']} since {w['since']} ({_weeks(w['weeks'])}), "
                f"score {w['score']:+.3f} as of {w['as_of']}")
    L = [head]
    if a.get("event"):
        L.append(f"  NEW THIS WEEK: the weekly signal turned {w['state']} at the {w['as_of']} close ({a['event']} signal).")
    if a.get("pending"):
        L.append(f"  PENDING: at the latest close the signal would be {a['pending']}. Only the week's last close confirms it.")
    elif a.get("near_flip"):
        L.append("  NEAR A FLIP: the move that would flip it is within one typical week's move.")
    L += signal_lines(a)
    if a.get("note"):
        L.append(f"  Note: {a['note']}")
    return "\n".join(L)


def btc_text(b: dict) -> str:
    w, n = b["weekly"], b["latest"]
    was = f", was {b['previous_week']} last week" if b["previous_week"] and b["previous_week"] != b["signal"] else ""
    sc_txt = "n/a" if w["score"] is None else f"{w['score']:+.3f}"
    L = [f"BITCOIN TREND SIGNAL: {b['signal']} ({'hold FBTC.TO' if b['signal'] == 'IN' else 'out, cash'}{was}) "
         f"— weekly score {sc_txt}, {w['state'] or 'n/a'} since {w['since'] or 'n/a'} ({_weeks(w['weeks'])})"]
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
    L += onchain_lines(b.get("onchain"))
    return "\n".join(L)


def _pctl(p: float | None) -> str:
    if p is None:
        return "n/a"
    n = round(p * 100)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix} percentile"


def onchain_lines(o: dict | None) -> list[str]:
    if not o:
        return []
    if o.get("error"):
        return [f"  ON-CHAIN: unavailable ({o['error']})"]
    zone = {"high": " — HIGH: top 10% of its history", "low": " — LOW: bottom 10% of its history"}
    L = [f"  ON-CHAIN (information only; data for {o['data_date']} UTC, {o['source']}):",
         f"    Realized price (average cost of all coins) US${o['realized_price_usd']:,.0f}; price US${o['price_usd']:,.0f} "
         f"is {pct(o['price_vs_realized'], 0)} vs it.",
         f"    MVRV {o['mvrv']:.2f} ({_pctl(o['mvrv_pct'])}{zone.get(o['mvrv_zone'], '')}), MVRV Z {o['mvrv_z']}, "
         f"NUPL {o['nupl']:+.2f}.",
         f"    Puell multiple {o['puell']:.2f} ({_pctl(o['puell_pct'])}{zone.get(o['puell_zone'], '')}); "
         f"hash rate 30d vs 60d {pct(o['hash_ribbon'])}{' (MINER STRESS)' if o['miner_stress'] else ''}; "
         f"active addresses 30d vs 1y {pct(o['active_addr_trend'], 0)}."]
    h = o.get("after_similar_mvrv") or {}
    parts = [f"{k[5:-1]} days later median {pct(v['median'], 0)}, up {v['share_up']:.0%} of the time "
             f"(range {pct(v['worst'], 0)} to {pct(v['best'], 0)})" for k, v in h.items() if k.startswith("next_")]
    if parts:
        L.append(f"    Earlier days at a similar MVRV percentile ({h['days']} days since {h['first'][:4]}, overlapping, few cycles): "
                 + "; ".join(parts) + " (US$).")
    s = o.get("shadow_early_entry")
    if s:
        L.append(f"    Shadow early-entry rule (IN also while MVRV is in the bottom 10%; paper-traded, not the signal): {s}.")
    return L


def _mark(r: dict) -> str:
    if r.get("stale"):
        return "STALE"
    if r["event"]:
        return r["event"]
    if r["pending"]:
        return f">{r['pending']}?"
    return "near" if r["near_flip"] else ""


def scan_text(sc: dict, rows: list[dict] | None = None) -> str:
    """The scan table. `rows` = a filtered subset of sc['rows'] (default: all)."""
    rows = sc["rows"] if rows is None else rows
    c = sc["counts"]
    L = [f"Scan — data as of {sc['data_as_of']} close (weekly signal from {sc['signal_week']}): {c['scanned']} assets, "
         f"{c['up']} UP, {c['down']} DOWN" + (f", {c['no_signal']} without a signal yet" if c["no_signal"] else ""),
         "\nSIGNAL EVENTS:"]
    L += events_lines(sc["events"], sc["signal_week"])
    L.append(f"\nTRENDS, best first ({len(rows)} shown). mark: BUY/SELL = turned this week, >UP?/>DOWN? = pending, near = close to a flip."
             "\n  [model] = one of the 13 backtested ETFs, [held] = in the model portfolio; everything else is the watchlist (information only)."
             "\n  record = following the signal on that asset alone vs holding it: return per year / max drawdown.")
    for r in rows:
        tag = "[held]" if r.get("in_model") else "[model]" if r["tier"] == "model" else ""
        if r["trend"] is None:
            why = _stale(r["stale"]) if r.get("stale") else "needs 12 months of history"
            L.append(f"  {r['ticker']:10s} {'n/a':>6s} {'':4s} {'':4s}  {_mark(r):7s} no signal ({why})  {tag:7s} {r['kind']:11s} {r['name']}")
            continue
        x = r["returns"]
        L.append(f"  {r['ticker']:10s} {r['score']:+.3f} {r['trend']:4s} {r['weeks']:3d}w  {_mark(r):7s} "
                 f"1m {pct(x['21d']):>6s} 3m {pct(x['63d']):>6s} 6m {pct(x['126d']):>6s} 12m {pct(x['252d']):>7s}  "
                 f"flip {pct(r['flip_move'], 0):>4s}  {r['indicators_up']}/{r['indicators_known']}  "
                 f"{record_text(r['record']):42s} {tag:7s} {r['kind']:11s} {r['name']}")
    if sc.get("stale"):
        L.append("\nStale prices: " + "; ".join(f"{t} ({_stale(v)})" for t, v in sc["stale"].items()))
    return "\n".join(L)


def _opp_row(r: dict) -> str:
    tag = "[held]" if r.get("in_model") else "[model]" if r["tier"] == "model" else ""
    ras = f"q={r['risk_adj_score']:+.2f}" if r.get("risk_adj_score") is not None else "q=n/a"
    rsi = f"RSI {r['rsi_14']:2.0f}" if r.get("rsi_14") is not None else "RSI --"
    fw = f"{r['flip_weeks']:.1f}w" if r.get("flip_weeks") is not None else ""
    fp = f" (${r['flip_price']})" if r.get("flip_price") is not None else ""
    rec = record_text(r.get("record"))
    return (f"  {r['ticker']:10s} {r['score']:+.3f} {ras:8s} {rsi:6s}  flip {pct(r['flip_move'], 0):>4s}{fp:10s} {fw:5s} "
            f"{r['indicators']:3s}  {rec:42s} {tag:7s} {r['name']}")


def opportunities_text(opp: dict) -> str:
    """Render actionable buy/sell opportunities from scan.opportunities + model status."""
    c = opp["counts"]
    L = [f"Opportunities — data as of {opp['data_as_of']} close (weekly signal from {opp['signal_week']}); "
         f"{c['scanned']} scanned ({c['up']} UP, {c['down']} DOWN)"]
    mod = opp.get("model")
    if mod:
        L.append(f"\nMODEL PORTFOLIO ({mod['follow']}): "
                 + ", ".join(f"{t} {w:.1%}".replace(".0%", "%") for t, w in mod["portfolio"].items()))
        if mod.get("changes"):
            L.append("  This week's model changes: " + "; ".join(change_text(ch) for ch in mod["changes"]))
        else:
            L.append("  No model portfolio changes this week.")
        if mod.get("held_by_buffer") or mod.get("next_in_line"):
            L.append(f"  Buffer: kept={', '.join(mod['held_by_buffer']) or 'none'}; "
                     f"next in line={', '.join(mod['next_in_line']) or 'none'}")

    sections = [
        ("fresh_buys", "FRESH CONFIRMED BUYS (turned UP at last week-end)"),
        ("fresh_sells", "FRESH CONFIRMED SELLS (turned DOWN at last week-end)"),
        ("pending_up", "PENDING BUYS (latest close > UP, awaiting Friday close)"),
        ("pending_down", "PENDING SELLS (latest close > DOWN, awaiting Friday close)"),
        ("pullback_setups", "PULLBACKS IN CONFIRMED UPTRENDS (UP, 4+/5 indicators, RSI <= 48; timing context)"),
        ("strongest_uptrends", "HIGHEST-QUALITY UPTRENDS (ranked by risk-adjusted score q = score/vol)"),
        ("early_turnarounds", "EARLY TURNAROUND RADAR (DOWN, fast 1/3m score UP, within 2.5 wks of flipping)"),
        ("extended_warnings", "EXTENDED / OVERBOUGHT UPTRENDS (RSI >= 75 or >35% above 200d; chasing risk)"),
    ]
    for key, title in sections:
        items = opp.get(key) or []
        if items:
            L.append(f"\n{title}:")
            for r in items:
                L.append(_opp_row(r))
    L.append(f"\nNote: {opp['note']}")
    return "\n".join(L)


def compare_text(comp: dict) -> str:
    """Side-by-side comparison table of requested assets."""
    L = [f"Comparison — data as of {comp['data_as_of']} close (weekly signal from {comp['signal_week']}):"]
    for a in comp["assets"]:
        w, n = a["weekly"], a["latest"]
        sc = "n/a" if w["score"] is None else f"{w['score']:+.3f}"
        ras = f"q={a['risk_adj_score']:+.2f}" if a.get("risk_adj_score") is not None else "q=n/a"
        rsi = f"RSI {a['rsi_14']:2.0f}" if a.get("rsi_14") is not None else "RSI --"
        r = w["returns"]
        L.append(f"  {a['ticker']:10s} {sc:>6s} {w['state'] or 'n/a':4s} {w['weeks']:3d}w  {ras:8s} {rsi:6s}  "
                 f"1m {pct(r['21d']):>6s} 3m {pct(r['63d']):>6s} 6m {pct(r['126d']):>6s} 12m {pct(r['252d']):>7s}  "
                 f"flip {pct(n['flip_move'], 0):>4s}  {a['indicators_up']}/{a['indicators_known']}  "
                 f"{record_text(a.get('record')):42s} {a['name']}")
    for q, err in (comp.get("errors") or {}).items():
        L.append(f"  {q}: ERROR ({err})")
    return "\n".join(L)
