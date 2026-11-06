"""Bitcoin pot: a fixed share of the portfolio that is either all FBTC.TO (trend up) or all CASH.TO.

Rule, checked once a week after Friday's close (same score the core strategy uses for every sleeve):
  IN  when Bitcoin's average 1/3/6/12-month return beats cash (score > 0), else OUT (cash).
No stop-loss: in the backtest 20%/25%/30% stops gave -5%, -8% and +7% CAGR vs no stop, i.e. noise.

The pot sits on top of the unchanged core strategy, which keeps its own 15% Bitcoin cap, so
total Bitcoin can reach pot share + 15% x (1 - pot share).

The live signal always shows the pot at its full share. In the backtest the pot is pulled
back to its share only when it drifts more than `band` away (not a small trade every Friday);
do the same when you rebalance by hand.

Monitoring between weekly checks is information only: the flip price says how far Bitcoin
would have to fall (or rise) for the score to cross zero. Acting on it mid-week would be a
different, untested strategy.
"""

import numpy as np
import pandas as pd

from .strategy import Params, scores

BAND = 0.10  # rebalance the pot back to its share only beyond +-10 points of drift


def score(tri: pd.DataFrame, p: Params = Params()) -> float | None:
    """Core momentum score for Bitcoin at the last row of `tri` (None if < 12 months of history)."""
    sc = scores(tri[["btc", "cash"]], p)
    return float(sc["btc"]) if "btc" in sc else None


def is_in(tri: pd.DataFrame, p: Params = Params()) -> bool:
    s = score(tri, p)
    return s is not None and s > 0


def flip_level(tri: pd.DataFrame, p: Params = Params()) -> float | None:
    """Bitcoin total-return index level at which the score would be exactly 0 today.

    score(P) = mean_l(P / P_l - 1 - cash_l)  is linear in today's level P, so
    P* = (1 + mean_l cash_l) / mean_l(1 / P_l).  Divide by today's level for a % move.
    """
    if len(tri) < max(p.lookbacks) + 1:
        return None
    b, c = tri["btc"], tri["cash"]
    past = np.array([b.iloc[-1 - lb] for lb in p.lookbacks])
    cash = np.array([c.iloc[-1] / c.iloc[-1 - lb] - 1 for lb in p.lookbacks])
    return float((1 + cash.mean()) / np.mean(1 / past))


def combine(core: pd.Series, on: bool, share: float, current_btc: float | None = None,
            band: float = BAND) -> pd.Series:
    """Account target = core strategy on (1 - share) + Bitcoin pot on `share`.

    `current_btc` = Bitcoin weight held now (whole account). If the pot is IN and was IN, and the
    holding is within `band` of where it should be, keep the current amount instead of trimming.
    """
    w = core.astype(float) * (1 - share)
    for k in ("btc", "cash"):
        if k not in w:
            w[k] = 0.0
    pot_btc = share if on else 0.0
    if on and current_btc is not None:
        held_pot = current_btc - w["btc"]  # part of the current holding that belongs to the pot
        if held_pot > 0 and abs(held_pot - share) <= band:
            pot_btc = min(held_pot, 1 - w.drop("cash").sum())
    w["btc"] += pot_btc
    w["cash"] = 1.0 - w.drop("cash").sum()
    return w


def core_view(current: pd.Series | None, share: float) -> pd.Series | None:
    """What the core strategy 'holds' once the pot's Bitcoin is set aside (for its hysteresis)."""
    if current is None or share <= 0:
        return current
    cur = current.copy()
    if "btc" in cur:
        cur["btc"] = max(0.0, float(cur["btc"]) - share)
    return cur


def status(tri: pd.DataFrame, price: float | None, p: Params = Params()) -> dict:
    """Snapshot for the CLI / MCP: signal, windows, flip price, trend and drawdown."""
    s = score(tri, p)
    b, c = tri["btc"], tri["cash"]
    lvl = float(b.iloc[-1])
    flip = flip_level(tri, p)
    ma200 = float(b.iloc[-200:].mean()) if len(b) >= 200 else None
    peak = float(b.iloc[-756:].max())  # ~3 years: covers the current cycle's high
    out = {
        "signal": None if s is None else ("IN" if s > 0 else "OUT"),
        "score": None if s is None else round(s, 4),
        "windows": {f"{lb}d": round(float(b.iloc[-1] / b.iloc[-1 - lb] - 1), 4)
                    for lb in p.lookbacks if len(b) > lb},
        "cash_windows": {f"{lb}d": round(float(c.iloc[-1] / c.iloc[-1 - lb] - 1), 4)
                         for lb in p.lookbacks if len(c) > lb},
        "flip_move": None if flip is None else round(flip / lvl - 1, 4),
        "vs_200d_avg": None if ma200 is None else round(lvl / ma200 - 1, 4),
        "from_3y_high": round(lvl / peak - 1, 4),
        "fbtc_price": price,
    }
    if flip is not None and price:
        out["flip_price_fbtc"] = round(price * flip / lvl, 2)
    return out
