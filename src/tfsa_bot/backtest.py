"""Day-by-day simulation with realistic frictions.

- Signal computed at the close of the rebalance day using only data up to that close.
- You place the trade the NEXT trading day; new weights start earning the day after that
  (i.e. one full day of lag, which is conservative for manual execution).
- Weights drift with the market between rebalances.
- Trading cost per sleeve traded = `cost_bps` (bid/ask spread + slippage; Wealthsimple has no
  commission) plus `fx_fee` on US-listed sleeves (CAD<->USD conversion, ~1.5% each way).
- `band`: skip a rebalance if the holdings are the same and no weight moved by more than this.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .data import total_return_index
from .strategy import Params, target_weights
from .universe import BY_KEY


@dataclass
class Result:
    equity: pd.Series
    weights: pd.DataFrame  # held weights per day
    trades: pd.DataFrame  # executed target weights on trade days
    turnover: pd.Series
    stats: dict = field(default_factory=dict)


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    rule = {"weekly": "W-FRI", "monthly": "ME", "quarterly": "QE"}[freq]
    return pd.DatetimeIndex(s.groupby(pd.Grouper(freq=rule)).last().dropna().values)


def run(
    rets: pd.DataFrame,
    weight_fn: Callable[[pd.DataFrame, pd.Series], pd.Series],
    freq: str = "weekly",
    cost_bps: float = 10.0,
    fx_fee: float = 0.015,
    band: float = 0.05,
    start: str | None = None,
    end: str | None = None,
) -> Result:
    tri = total_return_index(rets)
    r = rets.fillna(0.0)
    idx = rets.index
    reb = set(rebalance_dates(idx, freq))
    first_ok = idx[0] if start is None else idx[idx >= pd.Timestamp(start)][0]
    last_ok = idx[-1] if end is None else idx[idx <= pd.Timestamp(end)][-1]

    cols = list(rets.columns)
    cost = pd.Series(
        [cost_bps / 1e4 + (fx_fee if BY_KEY[c].usd else 0.0) for c in cols], index=cols
    )
    w = pd.Series(0.0, index=cols)
    w["cash"] = 1.0
    pending: pd.Series | None = None  # decided at close t, executed at close t+1
    eq, held, trade_rows, turn = [], [], {}, {}
    value = 1.0

    for d in idx:
        if d < first_ok or d > last_ok:
            continue
        # 1) market moves with yesterday's holdings
        day_ret = float((w * r.loc[d]).sum())
        value *= 1 + day_ret
        w = w * (1 + r.loc[d])
        w = w / w.sum()

        # 2) execute yesterday's decision at today's close
        if pending is not None:
            tgt = pending.reindex(cols).fillna(0.0)
            same_set = set(tgt[tgt > 1e-9].index) == set(w[w > 1e-9].index)
            dev = (tgt - w).abs()
            if not (same_set and dev.max() < band):
                to = float(dev.sum()) / 2  # one-way turnover
                value *= 1 - float((dev * cost).sum())
                w = tgt
                trade_rows[d] = tgt
                turn[d] = to
            pending = None

        # 3) decide at today's close
        if d in reb:
            pending = weight_fn(tri.loc[:d], w)

        eq.append((d, value))
        held.append(w.rename(d))

    equity = pd.Series(dict(eq))
    res = Result(
        equity=equity,
        weights=pd.DataFrame(held),
        trades=pd.DataFrame(trade_rows).T,
        turnover=pd.Series(turn, dtype=float),
    )
    res.stats = stats(res, rets["cash"].loc[equity.index])
    return res


def stats(res: Result, cash_ret: pd.Series) -> dict:
    eq = res.equity
    dr = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = dr.std() * np.sqrt(252)
    ex = dr - cash_ret.reindex(dr.index).fillna(0)
    sharpe = ex.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else np.nan
    dd = eq / eq.cummax() - 1
    yearly = eq.resample("YE").last().pct_change()
    yearly.iloc[0] = eq.resample("YE").last().iloc[0] / eq.iloc[0] - 1
    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "MaxDD": dd.min(),
        "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "WorstYr": yearly.min(),
        "Trades/yr": len(res.trades) / years,
        "Turnover/yr": res.turnover.sum() / years,
        "Cash%": res.weights["cash"].mean() if "cash" in res.weights else 0.0,
        "Final$1k": 1000 * eq.iloc[-1],
    }


# ---- benchmarks -----------------------------------------------------------

def fixed(weights: dict[str, float]) -> Callable[[pd.DataFrame, pd.Series], pd.Series]:
    def fn(tri: pd.DataFrame, current: pd.Series) -> pd.Series:
        w = pd.Series(0.0, index=tri.columns)
        for k, v in weights.items():
            w[k] = v
        return w

    return fn


# Rough shape of XEQT (all-equity) and XBAL (60/40) — the "do nothing" alternatives.
XEQT_LIKE = {"ca_eq": 0.25, "us_eq": 0.45, "intl": 0.25, "em": 0.05}
XBAL_LIKE = {**{k: v * 0.6 for k, v in XEQT_LIKE.items()}, "bonds": 0.4}
VFV_ONLY = {"us_eq": 1.0}


def strategy_fn(p: Params) -> Callable[[pd.DataFrame, pd.Series], pd.Series]:
    return lambda tri, current: target_weights(tri, p, current)
