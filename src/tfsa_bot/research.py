"""Backtest report: strategy vs. do-nothing benchmarks, robustness grid, sub-periods.

Run:  uv run tfsa-bot research
"""

from dataclasses import replace

import pandas as pd

from . import backtest as bt
from . import btc, signals
from .data import build_returns
from .strategy import Params, target_weights

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


def mix(core: pd.Series, on: bool, share: float, current_btc: float | None = None,
            band: float = 0.10) -> pd.Series:
    """Backtest only: core strategy on (1 - share) + Bitcoin slice on `share` following the signal.

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


def btc_pot_fn(p: Params, share: float, follow_trend: bool = True):
    """Core on (1 - share) + a Bitcoin slice on `share` that follows btc.is_in (or always holds)."""
    def fn(tri, current):
        core = target_weights(tri, p, core_view(current, share))
        on = btc.is_in(tri) if follow_trend else True
        return mix(core, on, share, float(current.get("btc", 0.0)))
    return fn


def btc_rule_fn(rule):
    """All-or-nothing Bitcoin vs cash, following `rule(tri) -> bool`."""
    def fn(tri, current):
        on = bool(rule(tri))
        return pd.Series({"btc": float(on), "cash": float(not on)})
    return fn


def indicator_rule(name: str):
    return lambda tri: {i["name"]: i["state"] for i in signals.indicators(tri, "btc")}[name] == "UP"


def halving_calendar_rule(tri) -> bool:
    """OUT during months 18-30 after a halving. The window was drawn AFTER looking at the three past
    cycles, so its backtest is a description of them, not a test."""
    d = tri.index[-1]
    past = [h for h in btc.HALVINGS if h <= d]
    return not past or not (18 <= (d - past[-1]).days / 30.44 < 30)


def btc_signal_study(rets: pd.DataFrame, start: str = "2015-09-01") -> None:
    """Section 8: is there a better Bitcoin signal than the core's score? Each rule runs alone on
    Bitcoin vs cash (weekly unless stated), then we look at how late each one was at the turns."""
    r = rets[["btc", "cash"]]
    rules = {
        "always Bitcoin": (lambda tri: True, "weekly"),
        "SIGNAL: trend score 1/3/6/12m": (indicator_rule("trend_score"), "weekly"),
        "same, checked daily": (indicator_rule("trend_score"), "daily"),
        "fast score 1/3m": (indicator_rule("fast_score"), "weekly"),
        "price > 100-day avg": (indicator_rule("above_100d"), "weekly"),
        "price > 200-day avg": (indicator_rule("above_200d"), "weekly"),
        "50-day > 200-day avg": (indicator_rule("ma50_above_ma200"), "weekly"),
        "halving calendar (hindsight)": (halving_calendar_rule, "weekly"),
    }
    rows = {}
    for name, (rule, freq) in rules.items():
        full = bt.run(r, btc_rule_fn(rule), freq=freq, start=start).stats
        rows[name] = {k: full[k] for k in ("CAGR", "Sharpe", "MaxDD", "Trades/yr", "Cash%")}
        for label, a, b in (("15-20", start, "2020-12-31"), ("21-now", "2021-01-01", None)):
            st = bt.run(r, btc_rule_fn(rule), freq=freq, start=a, end=b).stats
            rows[name][f"CAGR {label}"] = f"{st['CAGR']:.0%}"
            rows[name][f"MaxDD {label}"] = f"{st['MaxDD']:.0%}"
    print(table(rows), "\n")

    tri = (1 + r.fillna(0)).cumprod()
    b = tri["btc"]
    turns = {}
    for label, a, z, low in (("low 2018", "2018-06", "2019-06", True), ("high 2021", "2021-01", "2022-01", False),
                             ("low 2022", "2022-06", "2023-06", True), ("high 2025", "2024-06", None, False)):
        seg = b.loc[a:z]
        turns[f"{label} ({(seg.idxmin() if low else seg.idxmax()).date()})"] = (seg.idxmin() if low else seg.idxmax(), low)
    weeks = bt.rebalance_dates(tri.index, "weekly")
    late = {}
    for name, (rule, freq) in rules.items():
        if name == "always Bitcoin":
            continue
        dates = tri.index if freq == "daily" else weeks
        late[name] = {}
        for label, (d, low) in turns.items():
            hit = next((x for x in dates if x > d and rule(tri.loc[:x]) == low), None)  # IN after a low, OUT after a high
            late[name][label] = "-" if hit is None else f"{hit.date()} ({b.loc[hit] / b.loc[d] - 1:+.0%})"
        late[name]["now"] = "IN" if rule(tri) else "OUT"
    print("First IN after each low / first OUT after each high (and how far the price had already moved):")
    print(pd.DataFrame(late).T.to_string())
    print("The trend score gives up ~45-55% off the lows and ~25-40% off the highs: that is the price of waiting for\n"
          "confirmation. Faster rules catch the turn ~25 points sooner but have lower returns and deeper drawdowns\n"
          "overall; checking daily is within noise. The halving calendar looks best because it was fitted to these\n"
          "three cycles after the fact. Verdict: keep the score as the signal; show the rest as information.")


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
    print((held * 100).round(0).astype(int).astype(str).add("%").to_string(), "\n")

    print("=== 7. A Bitcoin slice following the Bitcoin signal, on top of the core (from 2015-09, first date with 12m of Bitcoin) ===")
    bs = "2015-09-01"
    rows = {"core only (pot off)": bt.run(rets, bt.strategy_fn(base), start=bs).stats}
    for share in (0.25, 0.5):
        rows[f"{share:.0%} pot, weekly trend rule"] = bt.run(rets, btc_pot_fn(base, share), start=bs).stats
        rows[f"{share:.0%} pot, always Bitcoin"] = bt.run(rets, btc_pot_fn(base, share, False), start=bs).stats
    rows.update(benchmarks(rets, bs))
    print(table(rows))
    print("Bitcoin's 2015-2021 gains (from a tiny base) dominate these numbers; don't expect them again.\n")

    print("=== 8. Bitcoin signal alternatives (Bitcoin or cash, from 2015-09) ===")
    btc_signal_study(rets)


if __name__ == "__main__":
    main()
