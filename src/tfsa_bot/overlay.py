"""The risk overlay. It can only REDUCE risk relative to the core strategy, never add it.

Inputs: core target weights, what's held now, a risk level, and per-ticker flags.
  - risk "high": no NEW positions this week (those slots go to cash); existing holdings
    follow the core (the core's own trend filter handles exits).
  - flag "avoid_new" on a sleeve: don't open it if not already held.
  - flag "reduce" on a sleeve: hold at most half of the core weight.
Every final weight is <= the core weight; the difference goes to cash.
"""

import pandas as pd

RANK = {"normal": 0, "elevated": 1, "high": 2}


def combine_levels(*levels: str | None) -> str:
    ls = [l for l in levels if l in RANK]
    return max(ls, key=RANK.get) if ls else "normal"


def apply(
    core: pd.Series,
    current: pd.Series | None,
    risk_level: str = "normal",
    flags: dict[str, str] | None = None,
) -> tuple[pd.Series, list[str]]:
    flags = flags or {}
    if "cash" not in core.index:
        core = pd.concat([core, pd.Series({"cash": 0.0})])
    cur =current.reindex(core.index).fillna(0.0) if current is not None else pd.Series(0.0, index=core.index)
    w = core.copy()
    notes = []
    for k in core.index:
        if k == "cash" or core[k] <= 0:
            continue
        new = cur[k] <= 1e-9
        if new and risk_level == "high":
            w[k] = 0.0
            notes.append(f"{k}: new entry blocked (market risk high)")
        elif new and flags.get(k) == "avoid_new":
            w[k] = 0.0
            notes.append(f"{k}: new entry blocked (news flag)")
        elif flags.get(k) == "reduce":
            w[k] = core[k] / 2
            notes.append(f"{k}: halved (news flag)")
    w["cash"] = 1.0 - w.drop("cash").sum()
    return w, notes
