"""The one strategy: trend-following momentum across asset classes, with a cash fallback.

Every rebalance:
  1. Score each sleeve = average total return over 1/3/6/12 months, MINUS what cash earned
     over the same windows (so "positive" means it beat a savings account).
  2. Hold the top N (default 4) sleeves with a positive score, 1/N each. If fewer qualify,
     the empty slots sit in cash. Bonds and gold compete like anything else.
  3. Hysteresis: a holding is kept while it still ranks within top N + hold_buffer.
  4. Cap crypto. Optional: inverse-volatility weights, portfolio volatility target.

This function is pure and only sees data up to "today", so the backtest and the live
signal run exactly the same code.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .universe import BY_KEY, SLEEVES


@dataclass(frozen=True)
class Params:
    lookbacks: tuple[int, ...] = (21, 63, 126, 252)
    top_n: int = 4
    weighting: str = "equal"  # equal | inv_vol  (equal was more robust in the grid)
    vol_window: int = 63
    target_vol: float | None = None  # e.g. 0.12 -> scale exposure down when risk is high
    include_crypto: bool = True
    markets: tuple[str, ...] = ("CA",)  # add "US" only with a USD account; FX fees ate the edge
    hold_buffer: int = 2  # keep a holding while it still ranks within top_n + hold_buffer
    usd_hurdle: float = 0.03  # a US-listed sleeve we don't hold must beat others by this (pays ~3% FX round trip)


def scores(tri: pd.DataFrame, p: Params) -> pd.Series:
    """Momentum score per sleeve at the last row of `tri` (total-return index incl. 'cash')."""
    need = max(p.lookbacks) + 1
    if len(tri) < need:
        return pd.Series(dtype=float)
    now = tri.iloc[-1]
    rel = []
    for lb in p.lookbacks:
        past = tri.iloc[-1 - lb]
        r = now / past - 1
        rel.append(r - r["cash"])
    sc = pd.concat(rel, axis=1).mean(axis=1).drop("cash")
    return sc.dropna()  # sleeves without a full 12m history are not eligible


def universe_keys(p: Params) -> list[str]:
    return [
        s.key
        for s in SLEEVES
        if (p.include_crypto or s.kind != "crypto") and s.market in p.markets
    ]


def target_weights(
    tri: pd.DataFrame, p: Params = Params(), current: pd.Series | None = None
) -> pd.Series:
    """Target portfolio weights (summing to 1, including 'cash') given history up to today.

    `current` = weights held right now (enables hysteresis and the USD hurdle).
    """
    keys = universe_keys(p)
    sc = scores(tri[keys + ["cash"]], p)
    w = pd.Series(0.0, index=keys + ["cash"])
    if sc.empty:
        w["cash"] = 1.0
        return w

    held = set()
    if current is not None:
        held = {k for k, v in current.items() if k != "cash" and v > 1e-9}
    for k in sc.index:
        if BY_KEY[k].usd and k not in held:
            sc[k] -= p.usd_hurdle

    ranked = list(sc[sc > 0].sort_values(ascending=False).index)
    chosen = [k for k in ranked[: p.top_n + p.hold_buffer] if k in held][: p.top_n]
    for k in ranked:
        if len(chosen) >= p.top_n:
            break
        if k not in chosen:
            chosen.append(k)
    picks = sc[chosen]
    if not picks.empty:
        rets = tri[picks.index].pct_change().iloc[-p.vol_window :]
        if p.weighting == "inv_vol":
            vol = rets.std() * np.sqrt(252)
            raw = 1 / vol.clip(lower=0.02)
        else:
            raw = pd.Series(1.0, index=picks.index)
        # each slot is worth 1/top_n; unfilled slots stay in cash
        sel = raw / raw.sum() * (len(picks) / p.top_n)

        for k in sel.index:
            sel[k] = min(sel[k], BY_KEY[k].max_weight)

        if p.target_vol:
            cov = rets.cov() * 252
            port_vol = float(np.sqrt(sel.values @ cov.values @ sel.values))
            if port_vol > p.target_vol:
                sel *= p.target_vol / port_vol

        w[sel.index] = sel
    w["cash"] = 1.0 - w.drop("cash").sum()
    return w
