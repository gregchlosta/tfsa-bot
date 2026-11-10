"""Bitcoin trend signal: IN (hold FBTC.TO) or OUT (cash), checked once a week after Friday's close.

Rule = the same score the core strategy uses for every ETF: IN when Bitcoin's average
1/3/6/12-month return beats cash (score > 0). `research` section 8 tested the alternatives
(faster scores, moving averages, checking daily): none beat it. Faster rules got in ~25 points
closer to the low but lost more to false starts. No stop-loss: 20%/25%/30% stops gave -5%, -8%
and +7% CAGR vs no stop, i.e. noise.

The bot doesn't size anything: how much goes into Bitcoin is decided outside it.

Everything besides the weekly signal is information only: the flip price (how far Bitcoin has to
move for the score to cross zero), the indicator panel and the cycle clock. The cycle clock
reports where we are after the last halving and what followed the same point in earlier cycles.
That is three cycles, so it is a description, not evidence; it never changes the signal.
"""

import pandas as pd

from . import signals
from .strategy import Params

# Block-reward halvings (UTC dates).
HALVINGS = tuple(pd.Timestamp(d) for d in ("2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"))


def score(tri: pd.DataFrame, p: Params = Params()) -> float | None:
    """Core momentum score for Bitcoin at the last row of `tri` (None if < 12 months of history)."""
    return signals.score(tri, "btc", p.lookbacks)


def is_in(tri: pd.DataFrame, p: Params = Params()) -> bool:
    s = score(tri, p)
    return s is not None and s > 0


def cycle_clock(tri: pd.DataFrame, horizons: tuple[int, ...] = (63, 126, 252)) -> dict | None:
    """Where Bitcoin is in the halving cycle, and what it did from the same day-count after
    earlier halvings (returns in CAD over the next ~3/6/12 months of trading days)."""
    b = tri["btc"].dropna()
    today = b.index[-1]
    past = [h for h in HALVINGS if h <= today]
    if not past:
        return None
    last = past[-1]
    days = (today - last).days
    cyc = b.loc[last:]
    high_d = cyc.idxmax()
    after = cyc.loc[high_d:]
    low_d = after.idxmin()
    out = {
        "last_halving": last.date().isoformat(),
        "days_since_halving": days,
        "cycle_high_date": high_d.date().isoformat(),
        "days_since_cycle_high": (today - high_d).days,
        "from_cycle_high": round(float(b.iloc[-1] / cyc.max() - 1), 4),
        "low_since_high_date": low_d.date().isoformat(),
        "low_since_high": round(float(after.min() / cyc.max() - 1), 4),
        "above_that_low": round(float(b.iloc[-1] / after.min() - 1), 4),
        "same_point_in_earlier_cycles": [],
        "note": "Three past cycles at most: a description, not evidence. It never changes the signal.",
    }
    for h in past[:-1]:
        t = h + pd.Timedelta(days=days)
        if t < b.index[0]:
            continue
        i = int(b.index.searchsorted(t))
        if i >= len(b):
            continue
        row = {
            "halving": h.date().isoformat(),
            "date": b.index[i].date().isoformat(),
            # only when our history covers that whole cycle
            "from_cycle_high": round(float(b.iloc[i] / b.loc[h:b.index[i]].max() - 1), 4) if b.index[0] <= h else None,
            "next": {f"{n}d": round(float(b.iloc[i + n] / b.iloc[i] - 1), 4) for n in horizons if i + n < len(b)},
        }
        out["same_point_in_earlier_cycles"].append(row)
    return out


def status(tri_w: pd.DataFrame, tri: pd.DataFrame, p: Params = Params(),
           close: float | None = None, usd_close: float | None = None) -> dict:
    """Snapshot for the CLI / MCP: the generic asset signal for Bitcoin plus the cycle clock.

    `close` = FBTC price and `usd_close` = BTC-USD price at tri's last row (for flip prices)."""
    out = signals.asset_signal(tri_w, tri, "btc", p, close)
    flip = out["latest"]["flip_move"]
    if usd_close and flip is not None:
        out["latest"]["btc_usd"] = round(float(usd_close))
        out["latest"]["flip_price_btc_usd"] = round(float(usd_close) * (1 + flip), -2)
    out["cycle"] = cycle_clock(tri)
    return out
