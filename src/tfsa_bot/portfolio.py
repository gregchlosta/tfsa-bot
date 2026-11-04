"""Your real Wealthsimple account, reconstructed from the trades and deposits you log."""

from collections import defaultdict
from datetime import date

import pandas as pd

from . import db
from .universe import BY_KEY

TICKER_TO_KEY = {s.ticker: k for k, s in BY_KEY.items()}


def normalize_ticker(ticker: str) -> str:
    """Accept how Wealthsimple shows tickers: 'ZEB' -> 'ZEB.TO', 'ETHX.B' -> 'ETHX-B.TO'."""
    t = ticker.strip().upper()
    for cand in (t, f"{t}.TO", f"{t.replace('.', '-')}.TO"):
        if cand in TICKER_TO_KEY:
            return cand
    return t


def record_trade(ticker: str, side: str, shares: float, price: float, day: str | None = None, note: str = "") -> str:
    if shares <= 0 or price <= 0:
        raise ValueError("shares and price must be positive")
    tk = normalize_ticker(ticker)
    db.add_trade(day or today(), tk, shares if side.upper() == "BUY" else -shares, price, note)
    return tk


def state(prices: pd.Series) -> dict:
    """Holdings, cash, and market value right now."""
    shares: dict[str, float] = defaultdict(float)
    deposits = sum(c["amount"] for c in db.cashflows())
    cash = deposits
    for t in db.trades():
        shares[t["ticker"]] += t["shares"]
        cash -= t["shares"] * t["price"]
    positions = []
    for tk, q in sorted(shares.items()):
        if abs(q) < 1e-6:
            continue
        px = prices.get(tk)
        px = None if px is None or pd.isna(px) else float(px)
        positions.append({"ticker": tk, "shares": round(q, 4), "price": px,
                          "value": round(q * px, 2) if px else 0.0,  # unknown ticker: no price, valued at 0
                          "in_strategy": tk in TICKER_TO_KEY})
    total = sum(p["value"] for p in positions) + cash
    return {
        "positions": positions, "cash": round(cash, 2), "total": round(total, 2),
        "net_deposits": round(deposits, 2), "gain": round(total - deposits, 2),
        "has_account": bool(positions) or abs(cash) > 0.01,
    }


def current_weights(st: dict) -> pd.Series | None:
    """Account as strategy weights (uninvested cash counts as the 'cash' sleeve)."""
    if not st["has_account"] or st["total"] <= 0:
        return None
    w = defaultdict(float)
    for p in st["positions"]:
        if p["in_strategy"]:
            w[TICKER_TO_KEY[p["ticker"]]] += p["value"] / st["total"]
    w["cash"] += st["cash"] / st["total"]
    return pd.Series(w)


def orders(target: pd.Series, st: dict, prices: pd.Series, capital: float, min_dollars: float) -> list[dict]:
    """Trades that move the account to `target`. Sells listed first so cash is there for buys.

    The 'cash' sleeve is held as CASH.TO, apart from a small buffer left as plain cash.
    """
    total = st["total"] if st["has_account"] else capital
    if total <= 0:
        return []
    have = defaultdict(float)
    for p in st["positions"]:
        have[p["ticker"]] += p["value"]
    out = []
    buffer = min(50.0, 0.02 * total)
    tickers = {BY_KEY[k].ticker: float(v) for k, v in target.items() if v > 1e-9}
    if "CASH.TO" in tickers:
        tickers["CASH.TO"] = max(0.0, tickers["CASH.TO"] - buffer / total)
    # strategy tickers we hold but no longer want
    for p in st["positions"]:
        if p["in_strategy"] and p["ticker"] not in tickers:
            tickers[p["ticker"]] = 0.0
    for tk, w in tickers.items():
        px = float(prices[tk])
        diff = w * total - have[tk]
        if abs(diff) < min_dollars:
            continue
        out.append({
            "side": "BUY" if diff > 0 else "SELL", "ticker": tk, "name": BY_KEY[TICKER_TO_KEY[tk]].name,
            "dollars": round(abs(diff), 2), "shares": round(abs(diff) / px, 3), "price": round(px, 2),
            "target_weight": round(w, 4),
        })
    out.sort(key=lambda o: (o["side"] != "SELL", -o["dollars"]))
    return out


def today() -> str:
    return date.today().isoformat()
