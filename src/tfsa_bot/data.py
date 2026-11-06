"""Download prices and build one CAD total-return index per sleeve, on the TSX calendar."""

import io
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from .universe import CASH, CORE_KEYS, FX_TICKER, SLEEVES, STRESS_TICKERS

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_CACHE = DATA_DIR / "raw_prices.pkl"
CASH_CACHE = DATA_DIR / "cad_tbill.pkl"
CACHE_MAX_AGE_H = 6


TZ = ZoneInfo("America/Toronto")
CLOSE_FINAL = (16, 30)  # Toronto time; matches engine.completed_returns


def _fresh(path: Path, now: datetime | None = None) -> bool:
    """Cache is fresh if younger than CACHE_MAX_AGE_H AND it wasn't saved before today's close while it's
    now after the close. Otherwise a pre-close snapshot would be treated as the completed daily bar."""
    if not path.exists():
        return False
    now = now or datetime.now(TZ)
    saved = datetime.fromtimestamp(path.stat().st_mtime, TZ)
    if (now - saved).total_seconds() >= CACHE_MAX_AGE_H * 3600:
        return False
    close = now.replace(hour=CLOSE_FINAL[0], minute=CLOSE_FINAL[1], second=0, microsecond=0)
    return not (now.weekday() < 5 and saved < close <= now)


def _all_tickers() -> list[str]:
    t = {FX_TICKER, CASH.ticker, *STRESS_TICKERS}
    for s in SLEEVES:
        t.add(s.ticker)
        if s.proxy:
            t.add(s.proxy)
    return sorted(t)


def load_raw(refresh: bool = False) -> pd.DataFrame:
    """Adjusted (dividend-inclusive) closes for every ticker we need, cached on disk."""
    if not refresh and _fresh(RAW_CACHE):
        return pd.read_pickle(RAW_CACHE)
    DATA_DIR.mkdir(exist_ok=True)
    df = yf.download(_all_tickers(), start="2000-01-01", auto_adjust=True, progress=False)["Close"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_pickle(RAW_CACHE)
    return df


def load_cad_tbill(refresh: bool = False) -> pd.Series:
    """Canadian 3-month T-bill yield (annual %, monthly) from FRED."""
    if not refresh and _fresh(CASH_CACHE):
        return pd.read_pickle(CASH_CACHE)
    sid = CASH.proxy.split(":", 1)[1]
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    with urllib.request.urlopen(url, timeout=30) as r:
        s = pd.read_csv(io.BytesIO(r.read()), index_col=0, parse_dates=True).iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").dropna()
    DATA_DIR.mkdir(exist_ok=True)
    s.to_pickle(CASH_CACHE)
    return s


def _splice(real: pd.Series, proxy_ret: pd.Series | None) -> pd.Series:
    """Daily returns: real ETF where it exists, proxy before that."""
    r_real = real.pct_change()
    first = real.first_valid_index()
    if proxy_ret is None or first is None:
        return r_real
    out = proxy_ret.copy()
    out.loc[out.index > first] = r_real.loc[r_real.index > first]
    return out


def build_returns(refresh: bool = False) -> pd.DataFrame:
    """Daily CAD returns per sleeve (+ 'cash'), indexed on TSX trading days.

    A sleeve is NaN before any history exists for it.
    """
    raw = load_raw(refresh)
    cal = raw["XIC.TO"].dropna().index  # TSX trading calendar
    px = raw.reindex(raw.index.union(cal)).ffill(limit=5).reindex(cal)  # align US/crypto/FX to TSX days
    fx = px[FX_TICKER]

    rets = {}
    for s in SLEEVES:
        proxy_ret = None
        if s.proxy:
            p = px[s.proxy] * (fx if s.proxy_fx else 1.0)
            proxy_ret = p.pct_change() - s.proxy_drag / 252
        real = px[s.ticker] * (fx if s.usd else 1.0)  # everything measured in CAD
        rets[s.key] = _splice(real, proxy_ret)

    tbill = load_cad_tbill(refresh).reindex(cal, method="ffill")
    rets["cash"] = _splice(px[CASH.ticker], tbill / 100 / 252)

    df = pd.DataFrame(rets, index=cal)
    df = df.replace([np.inf, -np.inf], np.nan)
    for k in ("bonds", "cash"):
        df[k] = _despike(df[k])
    start = df[CORE_KEYS].dropna().index[0]
    return df.loc[start:]


def _despike(r: pd.Series, z: float = 6.0) -> pd.Series:
    """Remove bad ticks in low-volatility series: a huge move immediately reversed the next day.

    Uses the next day's return, but only zeroes pairs that net to ~0, so it can't leak a trend.
    """
    sd = r.rolling(63, min_periods=20).std().shift(1)
    nxt = r.shift(-1)
    bad = (r.abs() > z * sd) & (r * nxt < 0) & ((r + nxt).abs() < 0.35 * r.abs())
    out = r.copy()
    out[bad] = 0.0
    out[bad.shift(1, fill_value=False)] = 0.0
    return out


def total_return_index(rets: pd.DataFrame) -> pd.DataFrame:
    """Cumulative growth of $1 per sleeve; NaN until a sleeve's history starts."""
    started = rets.notna().cummax()
    return (1 + rets.fillna(0)).cumprod().where(started)
