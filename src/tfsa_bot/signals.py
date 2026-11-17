"""Per-asset trend signals: pure functions of a total-return index (columns: asset keys + 'cash').

The confirmed signal for any asset is the strategy's own score at the last completed week:
UP when its average 1/3/6/12-month return beats cash. Everything else here is context for
reading that signal: how far the price is from flipping it, how long it has held, whether it
just changed (`flags`), what following it has done on this asset (`track_record`), and a small
panel of independent trend indicators. `research` section 8 backtests every indicator of the
panel on Bitcoin; none beat the score, the faster ones turn earlier but whipsaw more.
"""

import numpy as np
import pandas as pd

from . import backtest as bt
from .strategy import Params
from .universe import BY_KEY

FAST = (21, 63)  # lookbacks of the "early" score

# role: how to read an indicator next to the confirmed signal
ROLES = {
    "trend_score": "signal",   # the backtested rule
    "fast_score": "early",     # 1/3-month score: turns sooner, more false starts
    "above_100d": "early",
    "above_200d": "confirm",   # historically about as fast as the signal
    "ma50_above_ma200": "slow",
}


def score(tri: pd.DataFrame, key: str, lookbacks: tuple[int, ...]) -> float | None:
    """Same number as strategy.scores for one asset: mean over lookbacks of (asset return - cash return)."""
    a, c = tri[key], tri["cash"]
    if len(a) <= max(lookbacks):
        return None
    v = np.mean([a.iloc[-1] / a.iloc[-1 - lb] - c.iloc[-1] / c.iloc[-1 - lb] for lb in lookbacks])
    return None if np.isnan(v) else float(v)


def score_history(tri: pd.DataFrame, key: str, lookbacks: tuple[int, ...]) -> pd.Series:
    """The score on every day (NaN until the longest lookback has history)."""
    a, c = tri[key], tri["cash"]
    return sum(a / a.shift(lb) - c / c.shift(lb) for lb in lookbacks) / len(lookbacks)


def flip_level(tri: pd.DataFrame, key: str, lookbacks: tuple[int, ...]) -> float | None:
    """Index level at which the score would be exactly 0 today.

    score(P) = P * mean_l(1 / P_l) - mean_l(cash growth over l)  is linear in today's level P.
    """
    a, c = tri[key], tri["cash"]
    if len(a) <= max(lookbacks):
        return None
    inv = np.mean([1 / a.iloc[-1 - lb] for lb in lookbacks])
    cash = np.mean([c.iloc[-1] / c.iloc[-1 - lb] for lb in lookbacks])
    v = cash / inv
    return None if np.isnan(v) else float(v)


def indicators(tri: pd.DataFrame, key: str, p: Params = Params()) -> list[dict]:
    """Trend indicators at the last row of `tri`. Each: state UP/DOWN (None without enough history),
    its value, and `flip_move` = the price move today that would flip it (None if not price-driven)."""
    a = tri[key]
    price = float(a.iloc[-1])
    n_hist = int(a.notna().sum())
    out = []

    def add(name: str, value: float | None, flip: float | None = None) -> None:
        ok = value is not None and not np.isnan(value)
        out.append({
            "name": name, "role": ROLES[name],
            "state": ("UP" if value > 0 else "DOWN") if ok else None,
            "value": round(float(value), 4) if ok else None,
            "flip_move": round(flip / price - 1, 4) if ok and flip is not None else None,
        })

    add("trend_score", score(tri, key, p.lookbacks), flip_level(tri, key, p.lookbacks))
    add("fast_score", score(tri, key, FAST), flip_level(tri, key, FAST))
    for n in (100, 200):
        # price > average of the last n closes (today included)  <=>  price > average of the n-1 before today
        has = n_hist >= n
        add(f"above_{n}d", price / a.iloc[-n:].mean() - 1 if has else None, a.iloc[-n:-1].mean() if has else None)
    add("ma50_above_ma200", a.iloc[-50:].mean() / a.iloc[-200:].mean() - 1 if n_hist >= 200 else None)
    return out


def snapshot(tri: pd.DataFrame, key: str, p: Params = Params()) -> dict:
    """The signal at the last row of `tri`: state, score, the window returns behind it, flip move."""
    a = tri[key]
    sc = score(tri, key, p.lookbacks)
    flip = flip_level(tri, key, p.lookbacks)
    return {
        "as_of": tri.index[-1].date().isoformat(),
        "state": None if sc is None else ("UP" if sc > 0 else "DOWN"),
        "score": None if sc is None else round(sc, 4),
        "returns": _windows(a, p),
        "flip_move": None if flip is None else round(flip / float(a.iloc[-1]) - 1, 4),
    }


def weekly_states(tri_w: pd.DataFrame, key: str, p: Params = Params()) -> pd.Series:
    """The confirmed signal at every week-end with enough history: True = UP."""
    h = score_history(tri_w, key, p.lookbacks)
    return h.reindex(bt.rebalance_dates(tri_w.index, "weekly")).dropna() > 0


def _run_length(up: pd.Series) -> int:
    return int((up == up.iloc[-1])[::-1].astype(int).cumprod().sum())


def streak(tri_w: pd.DataFrame, key: str, p: Params = Params()) -> tuple[str | None, int]:
    """(week-end date the current weekly state began, number of weekly signals it has held)."""
    up = weekly_states(tri_w, key, p)
    if up.empty:
        return None, 0
    n = _run_length(up)
    return up.index[-n].date().isoformat(), n


def track_record(tri: pd.DataFrame, key: str, p: Params = Params(), cost_bps: float = 10.0) -> dict | None:
    """What following this asset's weekly signal on its own (the asset while UP, cash while DOWN) did,
    next to holding it throughout. Same rules as the backtest: decide at the week's last close, trade at
    the next close, `cost_bps` per leg. None with under a year of signals.

    It answers "has this signal been worth following on this asset?"; it is not a forecast."""
    t = tri[[key, "cash"]].loc[tri[key].first_valid_index():]
    up = weekly_states(t, key, p)
    if len(up) < 52:
        return None
    r = t.pct_change().fillna(0.0).loc[up.index[0]:]
    after = up.astype(float).reindex(r.index).ffill().shift(1).fillna(0.0)  # position once that day's trade is done
    pos = after.shift(1).fillna(0.0)  # position that earns the day's return
    cost = (after - pos).abs() * 2 * cost_bps / 1e4  # a switch trades two legs: the asset and cash
    eq = ((1 + pos * r[key] + (1 - pos) * r["cash"]) * (1 - cost)).cumprod()
    hold = (1 + r[key]).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25

    def cagr(e: pd.Series) -> float:
        return round(float(e.iloc[-1] ** (1 / years) - 1), 4)

    def max_dd(e: pd.Series) -> float:
        return round(float((e / e.cummax() - 1).min()), 4)

    return {
        "from": r.index[0].date().isoformat(), "years": round(years, 1),
        "signal_cagr": cagr(eq), "hold_cagr": cagr(hold),
        "signal_max_dd": max_dd(eq), "hold_max_dd": max_dd(hold),
        "flips_per_year": round(float((after != pos).sum()) / years, 1),
        "time_in": round(float(pos.mean()), 2),
    }


def asset_signal(tri_w: pd.DataFrame, tri: pd.DataFrame, key: str, p: Params = Params(),
                 close: float | None = None) -> dict:
    """Everything about one asset's trend.

    `tri_w` = history up to the last completed week (the confirmed signal); `tri` = up to the latest
    completed close (monitoring only). `close` = the asset's price at tri's last row, to turn the
    flip move into a price.
    """
    a = tri[key].dropna()
    lvl = float(a.iloc[-1])
    weekly = snapshot(tri_w, key, p)
    up = weekly_states(tri_w, key, p)
    n = _run_length(up) if len(up) else 0
    weekly["since"], weekly["weeks"] = (up.index[-n].date().isoformat(), n) if n else (None, 0)
    weekly["flipped"] = n == 1 and len(up) > 1  # the signal changed at the last week-end
    latest = snapshot(tri, key, p)
    if close and latest["flip_move"] is not None:
        latest["close"] = round(float(close), 2)
        latest["flip_price"] = round(float(close) * (1 + latest["flip_move"]), 2)
    ind = indicators(tri, key, p)
    known = [i for i in ind if i["state"]]
    move = float(a.pct_change().iloc[-63:].std() * np.sqrt(5)) if len(a) > 20 else None
    return {
        "weekly": weekly,
        "latest": latest,
        **flags(weekly, latest, move),
        "weekly_move": None if move is None or np.isnan(move) else round(move, 4),
        "cash_returns": _windows(tri["cash"], p),
        "vs_200d_avg": round(lvl / float(a.iloc[-200:].mean()) - 1, 4) if len(a) >= 200 else None,
        "from_3y_high": round(lvl / float(a.iloc[-756:].max()) - 1, 4),
        "indicators": ind,
        "indicators_up": sum(i["state"] == "UP" for i in known),
        "indicators_known": len(known),
    }


def flags(weekly: dict, latest: dict, weekly_move: float | None) -> dict:
    """What is new about a signal:
      event     BUY / SELL when the confirmed weekly signal turned UP / DOWN at the last week-end
      pending   the state at the latest close, when it disagrees with the weekly signal (only the
                week's last close confirms it)
      near_flip the latest close agrees, but the flip is within one typical week's move
    """
    w, n = weekly["state"], latest["state"]
    event = {"UP": "BUY", "DOWN": "SELL"}[w] if w and weekly.get("flipped") else None
    pending = n if w and n and n != w else None
    flip = latest.get("flip_move")
    near = bool(w and n == w and flip is not None and weekly_move and abs(flip) < weekly_move)
    return {"event": event, "pending": pending, "near_flip": near}


def _windows(s: pd.Series, p: Params) -> dict:
    out = {}
    for lb in p.lookbacks:
        v = s.iloc[-1] / s.iloc[-1 - lb] - 1 if len(s) > lb else np.nan
        out[f"{lb}d"] = None if np.isnan(v) else round(float(v), 4)
    return out


def signal_changes(prev: pd.Series, now: pd.Series, min_move: float = 0.05) -> list[dict]:
    """What moved between two model portfolios: BUY (new), SELL (gone), INCREASE/DECREASE (>= min_move).
    Sells come first, so the cash is there for the buys."""
    out = []
    for k in sorted(set(prev.index) | set(now.index)):
        a, b = float(prev.get(k, 0.0)), float(now.get(k, 0.0))
        if a < 1e-4 and b >= 1e-4:
            act = "BUY"
        elif a >= 1e-4 and b < 1e-4:
            act = "SELL"
        elif abs(b - a) >= min_move:
            act = "INCREASE" if b > a else "DECREASE"
        else:
            continue
        out.append({"action": act, "ticker": BY_KEY[k].ticker, "name": BY_KEY[k].name,
                    "from": round(a, 4), "to": round(b, 4)})
    out.sort(key=lambda c: ("SELL", "DECREASE", "INCREASE", "BUY").index(c["action"]))
    return out
