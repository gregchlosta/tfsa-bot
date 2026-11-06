"""Offline tests on synthetic data: strategy rules, overlay guarantees, orders, paper replay."""

import numpy as np
import pandas as pd
import pytest

from tfsa_bot import db, overlay, paper, portfolio
from tfsa_bot.strategy import Params, target_weights, universe_keys

KEYS = universe_keys(Params()) + ["cash"]


def make_tri(drifts: dict[str, float], days: int = 300) -> pd.DataFrame:
    """Total-return index where each sleeve grows at a fixed daily drift (default: flat)."""
    idx = pd.bdate_range("2024-01-01", periods=days)
    rng = np.random.default_rng(0)
    rets = pd.DataFrame({k: drifts.get(k, 0.0) + rng.normal(0, 1e-4, days) for k in KEYS}, index=idx)
    rets["cash"] = 0.0001
    return (1 + rets).cumprod()


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")


def test_holds_top4_equal_weight():
    tri = make_tri({"ca_banks": 0.003, "ca_energy": 0.0025, "gold": 0.002, "us_eq": 0.0015, "intl": 0.001})
    w = target_weights(tri)
    held = set(w[w > 0].index) - {"cash"}
    assert held == {"ca_banks", "ca_energy", "gold", "us_eq"}
    assert w["ca_banks"] == pytest.approx(0.25)
    assert w.sum() == pytest.approx(1.0)


def test_all_cash_when_nothing_beats_cash():
    w = target_weights(make_tri({}))
    assert w["cash"] == pytest.approx(1.0)


def test_crypto_capped():
    w = target_weights(make_tri({"btc": 0.01}))
    assert w["btc"] == pytest.approx(0.15)
    assert w["cash"] == pytest.approx(0.85)


def test_hysteresis_keeps_holding_ranked_just_outside_top4():
    tri = make_tri({"ca_banks": 0.003, "ca_energy": 0.0025, "gold": 0.002, "us_eq": 0.0015, "intl": 0.001})
    current = pd.Series({"intl": 0.25, "ca_banks": 0.25, "ca_energy": 0.25, "gold": 0.25})
    w = target_weights(tri, current=current)
    assert w["intl"] > 0 and w["us_eq"] == 0


def test_overlay_never_adds_risk():
    core = pd.Series({"ca_eq": 0.25, "us_eq": 0.25, "gold": 0.25, "btc": 0.15, "cash": 0.10})
    current = pd.Series({"ca_eq": 0.25, "us_eq": 0.25, "cash": 0.5})
    for risk in ("normal", "elevated", "high"):
        w, _ = overlay.apply(core, current, risk, {"gold": "avoid_new", "us_eq": "reduce"})
        assert (w.drop("cash") <= core.drop("cash") + 1e-12).all()
        assert w.sum() == pytest.approx(1.0)


def test_overlay_rules():
    core = pd.Series({"ca_eq": 0.25, "gold": 0.25, "btc": 0.25, "cash": 0.25})
    current = pd.Series({"ca_eq": 0.25, "cash": 0.75})
    w, _ = overlay.apply(core, current, "high", {"ca_eq": "reduce"})
    assert w["gold"] == 0 and w["btc"] == 0  # new entries blocked on high risk
    assert w["ca_eq"] == pytest.approx(0.125)  # existing holding halved by flag


def test_normalize_ticker():
    assert portfolio.normalize_ticker("zeb") == "ZEB.TO"
    assert portfolio.normalize_ticker("ETHX.B") == "ETHX-B.TO"
    assert portfolio.normalize_ticker("AAPL") == "AAPL"


def test_account_state_and_orders():
    prices = pd.Series({"ZEB.TO": 80.0, "XIC.TO": 50.0, "CASH.TO": 50.0, "XIT.TO": 60.0})
    db.add_cashflow("2026-01-01", 1000)
    portfolio.record_trade("ZEB", "BUY", 5, 80.0, "2026-01-02")
    portfolio.record_trade("XIT.TO", "BUY", 2, 50.0, "2026-01-02")
    portfolio.record_trade("MYSTERY", "BUY", 1, 10.0, "2026-01-02")  # unknown ticker, no price
    st = portfolio.state(prices)
    assert st["cash"] == pytest.approx(1000 - 400 - 100 - 10)
    assert st["total"] == pytest.approx(490 + 400 + 120)  # mystery valued at 0
    target = pd.Series({"ca_banks": 0.5, "ca_eq": 0.5, "cash": 0.0})
    orders = {(o["side"], o["ticker"]) for o in portfolio.orders(target, st, prices, 0, 25)}
    assert ("SELL", "XIT.TO") in orders  # strategy no longer wants it
    assert ("BUY", "XIC.TO") in orders
    assert all(o[1] != "MYSTERY" for o in orders)  # never trades things outside the strategy


def test_orders_zero_capital():
    st = {"has_account": False, "positions": [], "total": 0}
    assert portfolio.orders(pd.Series({"ca_eq": 1.0}), st, pd.Series({"XIC.TO": 50.0}), 0, 25) == []


def test_paper_replay_executes_next_day():
    idx = pd.bdate_range("2026-01-05", periods=5)
    rets = pd.DataFrame(0.0, index=idx, columns=KEYS)
    rets["ca_eq"] = [0.0, 0.10, 0.10, 0.0, 0.0]
    db.save_decision("core", "2026-01-05", {"ca_eq": 1.0})
    eq, w = paper.replay(rets, db.decisions("core"))
    # decided Jan 5, bought at Jan 6 close -> misses Jan 6's +10%, earns Jan 7's +10%.
    # Cost: two legs (sell CASH.TO, buy XIC) x 10 bps, same as the backtest.
    assert eq.iloc[-1] == pytest.approx(1.10 * (1 - 0.002))
    assert w["ca_eq"] == pytest.approx(1.0)


def test_api_key_read_from_zshrc(tmp_path, monkeypatch):
    from tfsa_bot import news
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rc = tmp_path / ".zshrc"
    rc.write_text('alias x=y\nexport ANTHROPIC_API_KEY="sk-test-123"  # mine\n')
    assert news.ensure_api_key(rc)
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-123"


def test_api_key_missing_rc(tmp_path, monkeypatch):
    from tfsa_bot import news
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not news.ensure_api_key(tmp_path / "nope")


def test_price_cache_stale_after_close(tmp_path):
    import os
    from datetime import datetime
    from tfsa_bot import data
    f = tmp_path / "c.pkl"
    f.write_text("x")
    wed = lambda h, m: datetime(2026, 11, 4, h, m, tzinfo=data.TZ)
    os.utime(f, (wed(15, 40).timestamp(),) * 2)  # saved before the close
    assert data._fresh(f, now=wed(16, 0))        # still intraday: reuse
    assert not data._fresh(f, now=wed(17, 0))    # after close: must re-download
    os.utime(f, (wed(16, 45).timestamp(),) * 2)  # saved after the close
    assert data._fresh(f, now=wed(18, 0))
    assert not data._fresh(f, now=wed(23, 0))    # older than max age


# --- Bitcoin pot -------------------------------------------------------------

def test_btc_signal_in_when_trending_out_when_flat():
    from tfsa_bot import btc
    assert btc.is_in(make_tri({"btc": 0.003}))
    assert not btc.is_in(make_tri({}))  # flat Bitcoin doesn't beat cash
    assert btc.score(make_tri({}, days=100)) is None  # under 12 months of history


def test_btc_flip_level_makes_score_zero():
    from tfsa_bot import btc
    tri = make_tri({"btc": 0.003})
    lvl = btc.flip_level(tri)
    assert lvl < tri["btc"].iloc[-1]  # trending up: it takes a drop to flip
    at_flip = tri.copy()
    at_flip.iloc[-1, at_flip.columns.get_loc("btc")] = lvl
    assert btc.score(at_flip) == pytest.approx(0.0, abs=1e-12)


def test_btc_combine_in_and_out():
    from tfsa_bot import btc
    core = pd.Series({"ca_eq": 0.25, "us_eq": 0.25, "btc": 0.15, "cash": 0.35})
    w = btc.combine(core, True, 0.5)
    assert w["btc"] == pytest.approx(0.5 + 0.075)
    assert w["ca_eq"] == pytest.approx(0.125)
    assert w.sum() == pytest.approx(1.0)
    w = btc.combine(core, False, 0.5)
    assert w["btc"] == pytest.approx(0.075)
    assert w["cash"] == pytest.approx(0.5 + 0.175)
    assert w.sum() == pytest.approx(1.0)


def test_btc_combine_band_avoids_small_weekly_trades():
    from tfsa_bot import btc
    core = pd.Series({"ca_eq": 0.5, "cash": 0.5})
    assert btc.combine(core, True, 0.5, current_btc=0.56)["btc"] == pytest.approx(0.56)  # drifted 6pts: keep
    assert btc.combine(core, True, 0.5, current_btc=0.65)["btc"] == pytest.approx(0.5)   # 15pts: trim
    assert btc.combine(core, True, 0.5, current_btc=0.0)["btc"] == pytest.approx(0.5)    # entering: full share
    assert btc.combine(core, False, 0.5, current_btc=0.56)["btc"] == 0.0                 # OUT: sell it all
    for cur in (0.0, 0.4, 0.56, 0.9):
        w = btc.combine(core, True, 0.5, current_btc=cur)
        assert w.sum() == pytest.approx(1.0) and (w >= -1e-12).all()


def test_btc_core_view_sets_pot_aside():
    from tfsa_bot import btc
    cur = pd.Series({"ca_eq": 0.3, "btc": 0.55, "cash": 0.15})
    assert btc.core_view(cur, 0.5)["btc"] == pytest.approx(0.05)
    assert btc.core_view(cur, 0.0)["btc"] == pytest.approx(0.55)
    assert btc.core_view(None, 0.5) is None
