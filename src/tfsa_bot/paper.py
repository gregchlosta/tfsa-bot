"""Forward test: replay recorded decisions on real prices (same execution rules as the backtest).

Tracks:
  core     - the backtested strategy, exactly as specified
  overlay  - core + news/sentiment check (the part that can't be backtested)
  bench    - XEQT-like buy-and-hold from the same start date
  btc      - the Bitcoin trend signal (FBTC or cash, weekly), vs holding Bitcoin throughout
The question this answers after a few months: does the news check earn its keep?
"""

import pandas as pd

from . import backtest as bt
from . import db

COST_BPS = 10.0


def replay(rets: pd.DataFrame, decs: list[dict]) -> tuple[pd.Series, pd.Series]:
    """Equity curve (starts at 1.0 on the first decision date) and current drifted weights."""
    cols = list(rets.columns)
    w = pd.Series(0.0, index=cols)
    w["cash"] = 1.0
    if not decs:
        return pd.Series(dtype=float), w
    by_date = {pd.Timestamp(d["as_of"]): pd.Series(d["weights"]).reindex(cols).fillna(0.0) for d in decs}
    start = min(by_date)
    r = rets.fillna(0.0).loc[start:]
    value, eq, pending = 1.0, {}, None
    for d in r.index:
        value *= 1 + float((w * r.loc[d]).sum())
        w = w * (1 + r.loc[d])
        w = w / w.sum()
        if pending is not None:
            value *= 1 - float((pending - w).abs().sum()) * COST_BPS / 1e4
            w, pending = pending, None
        if d in by_date:
            pending = by_date[d]
        eq[d] = value
    if pending is not None:  # decided at the last close, not executed yet
        w = pending
    return pd.Series(eq), w


def current_weights(rets: pd.DataFrame, track: str) -> pd.Series | None:
    decs = db.decisions(track)
    return replay(rets, decs)[1] if decs else None


def report(rets: pd.DataFrame) -> dict:
    out = {}
    curves = {}
    for track in ("core", "overlay"):
        eq, w = replay(rets, db.decisions(track))
        if len(eq):
            curves[track] = eq
            out[track] = {"since": eq.index[0].date().isoformat(), "return": round(eq.iloc[-1] - 1, 4),
                          "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
                          "holding": {k: round(v, 3) for k, v in w.items() if v > 1e-4}}
    if curves:
        start = min(c.index[0] for c in curves.values())
        b = bt.run(rets, bt.fixed(bt.XEQT_LIKE), freq="quarterly", start=str(start.date())).equity
        out["bench_xeqt_like"] = {"since": str(start.date()), "return": round(b.iloc[-1] / b.iloc[0] - 1, 4)}
    eq, w = replay(rets, db.decisions("btc"))
    if len(eq):
        hold = (1 + rets["btc"].fillna(0).loc[eq.index[0]:]).cumprod()
        out["btc_signal"] = {"since": eq.index[0].date().isoformat(), "return": round(eq.iloc[-1] - 1, 4),
                          "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
                          "holding": "FBTC" if w.get("btc", 0) > 0.5 else "cash",
                          "bitcoin_held_throughout": round(float(hold.iloc[-1] / hold.iloc[0] - 1), 4)}
    out["weeks_of_data"] = len(db.decisions("core"))
    out["verdict"] = (
        "Too early to judge (need ~12+ weeks)." if out["weeks_of_data"] < 12
        else "Compare core vs overlay: switch config 'follow' to 'overlay' only if it is ahead on return AND drawdown."
    )
    return out
