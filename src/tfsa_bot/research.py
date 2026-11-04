"""Backtest report: strategy vs. do-nothing benchmarks, robustness grid, sub-periods.

Run:  uv run tfsa-bot research
"""

from dataclasses import replace

import pandas as pd

from . import backtest as bt
from .data import build_returns
from .strategy import Params

START = "2006-01-01"  # leaves 1y warm-up after data starts (late 2004)
FMT = {
    "CAGR": "{:.1%}", "Vol": "{:.1%}", "Sharpe": "{:.2f}", "MaxDD": "{:.1%}", "Calmar": "{:.2f}",
    "WorstYr": "{:.1%}", "Trades/yr": "{:.1f}", "Turnover/yr": "{:.1f}", "Cash%": "{:.0%}",
    "Final$1k": "{:,.0f}",
}


def table(rows: dict[str, dict]) -> str:
    df = pd.DataFrame(rows).T
    for c, f in FMT.items():
        if c in df:
            df[c] = df[c].map(lambda v, f=f: f.format(v))
    return df.to_string()


def benchmarks(rets, start=START, end=None) -> dict[str, dict]:
    kw = dict(freq="quarterly", start=start, end=end)
    return {
        "BENCH XEQT-like (100% equity)": bt.run(rets, bt.fixed(bt.XEQT_LIKE), **kw).stats,
        "BENCH XBAL-like (60/40)": bt.run(rets, bt.fixed(bt.XBAL_LIKE), **kw).stats,
        "BENCH VFV only (S&P 500)": bt.run(rets, bt.fixed(bt.VFV_ONLY), **kw).stats,
    }


def main() -> None:
    rets = build_returns()
    base = Params()
    pd.set_option("display.width", 250)

    print(f"Data {rets.index[0].date()} -> {rets.index[-1].date()}, backtest from {START}\n")

    print("=== 1. Strategy variants vs benchmarks (full period) ===")
    rows = {
        "DEFAULT (CA-listed, top4 equal, weekly)": bt.run(rets, bt.strategy_fn(base), start=START).stats,
        "+ US-listed, paying 1.5% FX": bt.run(rets, bt.strategy_fn(replace(base, markets=("CA", "US"))), start=START).stats,
        "+ US-listed, USD account (no FX)": bt.run(rets, bt.strategy_fn(replace(base, markets=("CA", "US"), usd_hurdle=0)), start=START, fx_fee=0).stats,
        "no crypto": bt.run(rets, bt.strategy_fn(replace(base, include_crypto=False)), start=START).stats,
        "monthly rebalance": bt.run(rets, bt.strategy_fn(base), freq="monthly", start=START).stats,
        "vol target 12%": bt.run(rets, bt.strategy_fn(replace(base, target_vol=0.12)), start=START).stats,
    }
    rows.update(benchmarks(rets))
    print(table(rows), "\n")

    print("=== 2. Robustness grid (CA-listed, weekly) — want MOST cells to beat benchmarks ===")
    grid = {}
    for top_n in (1, 2, 3, 4, 5):
        for weighting in ("equal", "inv_vol"):
            for buf in (0, 2):
                p = replace(base, top_n=top_n, weighting=weighting, hold_buffer=buf)
                s = bt.run(rets, bt.strategy_fn(p), start=START).stats
                grid[f"top{top_n} {weighting:7s} buf{buf}"] = s
    print(table(grid), "\n")

    print("=== 3. Lookback sensitivity ===")
    lbs = {
        "fast  (1,3m)": (21, 63),
        "mid   (3,6m)": (63, 126),
        "slow  (6,12m)": (126, 252),
        "blend (1,3,6,12m)": (21, 63, 126, 252),
    }
    print(table({k: bt.run(rets, bt.strategy_fn(replace(base, lookbacks=v)), start=START).stats for k, v in lbs.items()}), "\n")

    print("=== 4. Sub-periods (out-of-sample check: does it work in both halves?) ===")
    for a, b in (("2006-01-01", "2015-12-31"), ("2016-01-01", None)):
        rows = {"DEFAULT": bt.run(rets, bt.strategy_fn(base), start=a, end=b).stats}
        rows.update(benchmarks(rets, a, b))
        print(f"--- {a[:4]} to {(b or 'now')[:4]} ---")
        print(table(rows), "\n")

    print("=== 5. Calendar-year returns ===")
    res = bt.run(rets, bt.strategy_fn(base), start=START)
    bench = bt.run(rets, bt.fixed(bt.XEQT_LIKE), freq="quarterly", start=START)
    yr = pd.DataFrame({
        "strategy": res.equity.resample("YE").last().pct_change(),
        "XEQT-like": bench.equity.resample("YE").last().pct_change(),
    }).dropna()
    yr.index = yr.index.year
    print((yr * 100).round(1).to_string(), "\n")

    print("=== 6. How often each asset was held (default) ===")
    held = (res.weights > 1e-6).mean().sort_values(ascending=False)
    print((held * 100).round(0).astype(int).astype(str).add("%").to_string())


if __name__ == "__main__":
    main()
