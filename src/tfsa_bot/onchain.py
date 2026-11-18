"""Bitcoin on-chain readings: information next to the Bitcoin signal, never a signal.

Source: CoinMetrics Community API (free, no key, daily from 2010-07). Only metrics computed from
the chain itself are used: market cap / realized cap (MVRV), miner issuance (Puell multiple), hash
rate and active addresses. Exchange balances and flows are left out on purpose: they depend on
address labels that get revised after the fact, so a backtest on them would see the future.

Point in time: a value stamped UTC day d is complete at 00:00 UTC d+1, which is after the TSX
close of day d, so it is first used on the next trading day (`on_calendar`). Percentiles rank a
reading against the history known at that date only (expanding window), never the full history.

`research` section 11 backtested these readings as changes to the Bitcoin signal. Cutting the
position when valuations are high or miners are stressed was noise. Entering early when MVRV is in
the bottom 10% of its history was ahead, but on two bottoms only (late 2018, late 2022), so it runs
as a shadow paper track (`early_entry`, track `btc_onchain`) and the signal stays the trend score.
"""

import json
import urllib.request

import numpy as np
import pandas as pd

from .data import DATA_DIR, _fresh

URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = ("PriceUSD", "CapMrktCurUSD", "CapMVRVCur", "IssTotUSD", "HashRate", "AdrActCnt")
CACHE = DATA_DIR / "onchain.pkl"
HIGH, LOW = 0.90, 0.10  # percentile zones, fixed before looking at any result
MAX_AGE_DAYS = 4        # older than this = stale (CoinMetrics publishes daily)


def fetch() -> pd.DataFrame:
    """Full daily history of METRICS for BTC (UTC dates, floats)."""
    url = (f"{URL}?assets=btc&metrics={','.join(METRICS)}&frequency=1d&page_size=10000"
           "&start_time=2009-01-01")
    rows = []
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "tfsa-bot"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
        rows += d["data"]
        url = d.get("next_page_url")
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    return df.reindex(columns=list(METRICS)).apply(pd.to_numeric, errors="coerce").dropna(how="all")


def load(refresh: bool = False) -> pd.DataFrame:
    if not refresh and _fresh(CACHE):
        return pd.read_pickle(CACHE)
    df = fetch()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_pickle(CACHE)
    return df


def _pct_rank(s: pd.Series, min_periods: int = 365 * 2) -> pd.Series:
    """Percentile of each value within the history up to and including that day."""
    return s.expanding(min_periods=min_periods).rank(pct=True)


def features(raw: pd.DataFrame) -> pd.DataFrame:
    """Daily on-chain features on UTC dates, each computed from data up to that date only."""
    mvrv = raw["CapMVRVCur"]
    mc = raw["CapMrktCurUSD"]
    rc = mc / mvrv
    iss = raw["IssTotUSD"]
    hr = raw["HashRate"]
    adr = raw["AdrActCnt"].where(raw["AdrActCnt"] > 0)
    f = pd.DataFrame({
        "price_usd": raw["PriceUSD"],
        "realized_price_usd": raw["PriceUSD"] / mvrv,
        "mvrv": mvrv,
        # (market cap - realized cap) / std of market cap so far; the classic version uses the whole
        # history's std, which looks ahead
        "mvrv_z": (mc - rc) / mc.expanding(min_periods=365).std(),
        "puell": iss / iss.rolling(365, min_periods=365).mean(),
        # 30-day vs 60-day average hash rate: below zero = miners switching off (capitulation)
        "hash_ribbon": hr.rolling(30).mean() / hr.rolling(60).mean() - 1,
        "active_addr_trend": adr.rolling(30).mean() / adr.rolling(365, min_periods=365).mean() - 1,
    })
    f["mvrv_pct"] = _pct_rank(f["mvrv"])
    f["puell_pct"] = _pct_rank(f["puell"])
    return f


def on_calendar(feat: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Features as known at each trading day's close: UTC day d is first used on day d+1."""
    f = feat.copy()
    f.index = f.index + pd.Timedelta(days=1)
    return f.reindex(f.index.union(cal)).ffill(limit=MAX_AGE_DAYS).reindex(cal)


def zone(p: float | None) -> str | None:
    if p is None or np.isnan(p):
        return None
    return "high" if p >= HIGH else "low" if p <= LOW else "middle"


def _r(v, n: int = 4):
    return None if v is None or pd.isna(v) else round(float(v), n)


def history_at_same_level(feat: pd.DataFrame, pct: float, width: float = 0.05,
                          horizons: tuple[int, ...] = (182, 365)) -> dict | None:
    """What Bitcoin did (USD) after earlier days whose MVRV percentile (as known then) was within
    `width` of today's. Overlapping days from few cycles: a description, not a forecast."""
    p = feat["price_usd"]
    near = feat.index[(feat["mvrv_pct"] - pct).abs() <= width]
    out = {"days": int(len(near))}
    if not len(near):
        return None
    out["first"] = near[0].date().isoformat()
    for h in horizons:
        fwd = (p.shift(-h, freq="D").reindex(near) / p.reindex(near) - 1).dropna()
        if len(fwd) < 30:
            continue
        out[f"next_{h}d"] = {"median": _r(fwd.median()), "share_up": _r((fwd > 0).mean(), 2),
                             "worst": _r(fwd.min()), "best": _r(fwd.max()), "days": int(len(fwd))}
    return out


def known_at(feat: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series | None:
    """The latest feature row known at the close of `as_of` (UTC day < as_of), None if stale."""
    f = feat.loc[:pd.Timestamp(as_of).normalize() - pd.Timedelta(days=1)].dropna(subset=["mvrv"])
    if f.empty or (pd.Timestamp(as_of).normalize() - f.index[-1]).days > MAX_AGE_DAYS:
        return None
    return f.iloc[-1]


def early_entry(signal_in: bool, mvrv_pct: float | None) -> bool:
    """SHADOW rule, not the signal: the Bitcoin signal, or IN early while MVRV is in the bottom 10% of
    its history so far. Ahead in the backtest (research section 11) but on two bottoms only (late 2018
    and late 2022), so it is paper-traded as the `btc_onchain` track until a new cycle tests it."""
    return bool(signal_in) or (mvrv_pct is not None and not np.isnan(mvrv_pct) and mvrv_pct <= LOW)


def panel(feat: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """On-chain readings for the report, as known at the close of `as_of`."""
    row = known_at(feat, as_of)
    if row is None:
        last = feat["mvrv"].last_valid_index()
        return {"error": f"on-chain data is stale (last {last.date() if last is not None else 'none'})"}
    feat = feat.loc[:row.name]
    mv_pct = _r(row["mvrv_pct"], 3)
    hr = _r(row["hash_ribbon"])
    return {
        "data_date": row.name.date().isoformat(),
        "price_usd": _r(row["price_usd"], 0),
        "realized_price_usd": _r(row["realized_price_usd"], 0),
        "price_vs_realized": _r(row["price_usd"] / row["realized_price_usd"] - 1),
        "mvrv": _r(row["mvrv"], 3),
        "mvrv_pct": mv_pct,
        "mvrv_zone": zone(mv_pct),
        "mvrv_z": _r(row["mvrv_z"], 2),
        "nupl": _r(1 - 1 / row["mvrv"], 3),
        "puell": _r(row["puell"], 3),
        "puell_pct": _r(row["puell_pct"], 3),
        "puell_zone": zone(row["puell_pct"]),
        "hash_ribbon": hr,
        "miner_stress": None if hr is None else hr < 0,
        "active_addr_trend": _r(row["active_addr_trend"]),
        "after_similar_mvrv": None if mv_pct is None else history_at_same_level(feat, mv_pct),
        "source": "CoinMetrics Community API",
        "note": ("Information only; the Bitcoin signal is unchanged. Percentiles compare with the history known "
                 "at each date. Valuation peaks have been lower each cycle, so 'high' may not come again."),
    }
