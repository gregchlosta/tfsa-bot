"""Market stress gauge from data only (so it CAN be backtested).

Each condition that fires adds one point:
  - VIX above 30 (fear is high)
  - VIX up more than 40% in a week (fear is spiking)
  - Junk bonds falling vs. Treasuries by >3% in a month (credit stress)
  - S&P 500 more than 10% below its 1-year high
  - TSX below its 200-day average

0 -> normal, 1-2 -> elevated, 3+ -> high
"""

import pandas as pd

LEVELS = ["normal", "elevated", "high"]


def stress_table(raw: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    px = raw.reindex(raw.index.union(cal)).ffill(limit=5).reindex(cal)
    vix, spy, xic = px["^VIX"], px["SPY"], px["XIC.TO"]
    credit = px["HYG"] / px["IEF"]
    t = pd.DataFrame(index=cal)
    t["vix_high"] = vix > 30
    t["vix_spike"] = vix / vix.shift(5) - 1 > 0.40
    t["credit_stress"] = credit / credit.shift(21) - 1 < -0.03  # NaN before 2007 -> False
    t["spx_drawdown"] = spy / spy.rolling(252, min_periods=60).max() - 1 < -0.10
    t["tsx_downtrend"] = xic < xic.rolling(200, min_periods=100).mean()
    t["points"] = t.sum(axis=1)
    t["level"] = pd.cut(t["points"], [-1, 0, 2, 99], labels=LEVELS).astype(str)
    t["vix"] = vix
    return t


def snapshot(raw: pd.DataFrame, cal: pd.DatetimeIndex) -> dict:
    t = stress_table(raw, cal)
    row = t.iloc[-1]
    fired = [c for c in ("vix_high", "vix_spike", "credit_stress", "spx_drawdown", "tsx_downtrend") if row[c]]
    return {
        "as_of": t.index[-1].date().isoformat(),
        "level": row["level"],
        "points": int(row["points"]),
        "fired": fired,
        "vix": round(float(row["vix"]), 1),
    }
