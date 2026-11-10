"""Offline tests on synthetic data: strategy rules, overlay guarantees, signal changes, paper replay."""

import numpy as np
import pandas as pd
import pytest

from tfsa_bot import btc, db, overlay, paper, signals
from tfsa_bot.strategy import Params, scores, target_weights, universe_keys

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


def test_signal_changes():
    from tfsa_bot.signals import signal_changes
    prev = pd.Series({"ca_eq": 0.25, "ca_banks": 0.25, "gold": 0.25, "cash": 0.25})
    now = pd.Series({"ca_eq": 0.25, "gold": 0.125, "btc": 0.5, "cash": 0.125})
    ch = {c["ticker"]: c["action"] for c in signal_changes(prev, now)}
    assert ch == {"ZEB.TO": "SELL", "CGL-C.TO": "DECREASE", "CASH.TO": "DECREASE", "FBTC.TO": "BUY"}
    assert signal_changes(prev, prev + 0.01) == []  # small drifts aren't changes
    order = [c["action"] for c in signal_changes(prev, now)]
    assert order.index("SELL") < order.index("BUY")  # sells first, so the cash is there


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


# --- per-asset signals ----------------------------------------------------------

LB = Params().lookbacks
UP5 = {"ca_banks": 0.003, "ca_energy": 0.0025, "gold": 0.002, "us_eq": 0.0015, "intl": 0.001}


def test_signal_score_is_the_strategy_score():
    tri = make_tri(UP5)
    sc = scores(tri, Params())
    for k in sc.index:
        assert signals.score(tri, k, LB) == pytest.approx(sc[k], abs=1e-12)
        assert signals.score_history(tri, k, LB).iloc[-1] == pytest.approx(sc[k], abs=1e-12)
    assert signals.score(make_tri({}, days=100), "gold", LB) is None  # under 12 months of history


def test_flip_level_makes_score_zero():
    tri = make_tri({"btc": 0.003})
    lvl = signals.flip_level(tri, "btc", LB)
    assert lvl < tri["btc"].iloc[-1]  # trending up: it takes a drop to flip
    at_flip = tri.copy()
    at_flip.iloc[-1, at_flip.columns.get_loc("btc")] = lvl
    assert signals.score(at_flip, "btc", LB) == pytest.approx(0.0, abs=1e-12)


def test_indicators_agree_in_a_steady_uptrend_and_flip_moves_are_exact():
    tri = make_tri({"gold": 0.002})
    ind = {i["name"]: i for i in signals.indicators(tri, "gold")}
    assert set(ind) == set(signals.ROLES) and all(i["state"] == "UP" for i in ind.values())
    assert ind["ma50_above_ma200"]["flip_move"] is None  # not flippable by today's price
    for name in ("trend_score", "fast_score", "above_100d", "above_200d"):
        moved = tri.copy()
        moved.iloc[-1, moved.columns.get_loc("gold")] *= 1 + ind[name]["flip_move"]
        again = {i["name"]: i for i in signals.indicators(moved, "gold")}
        assert again[name]["value"] == pytest.approx(0.0, abs=2e-4)  # flip_move is rounded to 4 decimals
    flat = {i["name"]: i["state"] for i in signals.indicators(make_tri({}, days=150), "gold")}
    assert flat["trend_score"] is None and flat["above_200d"] is None and flat["above_100d"] is not None


def test_streak_counts_weeks_since_the_last_flip():
    idx = pd.bdate_range("2023-01-02", periods=500)
    r = np.where(np.arange(500) < 440, 0.002, -0.02)  # long rise, then 60 days of falling
    tri = pd.DataFrame({"x": (1 + r).cumprod(), "cash": 1.0}, index=idx)
    since, weeks = signals.streak(tri, "x")
    hist = signals.score_history(tri, "x", LB)
    flip_day = hist[hist > 0].index[-1]
    assert 0 < weeks < 12 and pd.Timestamp(since) > flip_day
    assert pd.Timestamp(since) - flip_day <= pd.Timedelta(days=7)
    assert signals.streak(tri.iloc[:440], "x")[1] > 30  # before the fall: UP since history allows a score


def test_asset_signal_weekly_vs_latest_and_flip_price():
    tri = make_tri({"gold": 0.002})
    a = signals.asset_signal(tri.iloc[:-3], tri, "gold", close=50.0)
    assert a["weekly"]["as_of"] == tri.index[-4].date().isoformat()
    assert a["latest"]["as_of"] == tri.index[-1].date().isoformat()
    assert a["weekly"]["state"] == "UP" and a["weekly"]["weeks"] >= 1
    assert a["latest"]["flip_price"] == pytest.approx(50.0 * (1 + a["latest"]["flip_move"]), abs=0.01)
    assert a["indicators_up"] == a["indicators_known"] == 5
    assert a["latest"]["returns"]["252d"] > a["cash_returns"]["252d"]


# --- Bitcoin signal -------------------------------------------------------------

def test_btc_signal_in_when_trending_out_when_flat():
    assert btc.is_in(make_tri({"btc": 0.003}))
    assert not btc.is_in(make_tri({}))  # flat Bitcoin doesn't beat cash
    assert btc.score(make_tri({}, days=100)) is None  # under 12 months of history


def test_btc_status_adds_usd_flip_price_and_cycle():
    tri = make_tri({"btc": 0.003})
    s = btc.status(tri, tri, close=40.0, usd_close=100_000.0)
    assert s["weekly"]["state"] == "UP"
    assert s["latest"]["flip_price_btc_usd"] == pytest.approx(100_000 * (1 + s["latest"]["flip_move"]), abs=50)
    assert s["cycle"]["last_halving"] == "2024-04-20"  # make_tri ends in early 2025


def test_btc_cycle_clock_compares_same_day_count_in_earlier_cycles():
    idx = pd.bdate_range("2016-01-01", "2025-06-30")
    lvl = pd.Series(np.linspace(1.0, 10.0, len(idx)), index=idx)  # steady climb: today is always the high
    c = btc.cycle_clock(pd.DataFrame({"btc": lvl, "cash": 1.0}))
    assert c["last_halving"] == "2024-04-20"
    assert c["days_since_halving"] == (idx[-1] - pd.Timestamp("2024-04-20")).days
    assert c["from_cycle_high"] == 0 and c["above_that_low"] == 0
    rows = {e["halving"]: e for e in c["same_point_in_earlier_cycles"]}
    assert set(rows) == {"2016-07-09", "2020-05-11"}  # 2012: before our history
    e = rows["2020-05-11"]
    i = idx.searchsorted(pd.Timestamp("2020-05-11") + pd.Timedelta(days=c["days_since_halving"]))
    assert e["date"] == idx[i].date().isoformat()
    assert e["next"]["252d"] == pytest.approx(lvl.iloc[i + 252] / lvl.iloc[i] - 1, abs=1e-4)
    assert e["from_cycle_high"] == 0
    early = btc.cycle_clock(pd.DataFrame({"btc": lvl.loc[:"2016-06-01"], "cash": 1.0}))
    assert early["last_halving"] == "2012-11-28" and early["same_point_in_earlier_cycles"] == []


def test_bitcoin_notification_states_map_to_in_out(monkeypatch):
    from tfsa_bot import engine
    sent = []
    monkeypatch.setattr(engine.notify, "send", lambda t, b: sent.append(t))
    base = {"changes": [], "new_week": False, "news": None, "news_fresh": False}
    b = {"signal": "OUT", "previous_week": "OUT", "new_week": False, "latest": {"state": "UP"}}
    engine._maybe_notify({**base, "bitcoin": b}, None, {})
    assert sent == ["Bitcoin heads-up"]
    engine._maybe_notify({**base, "bitcoin": b}, None, {"bitcoin": {"latest": {"state": "UP"}}})  # already told
    engine._maybe_notify({**base, "bitcoin": {**b, "latest": {"state": "DOWN"}}}, None, {})       # agrees
    assert sent == ["Bitcoin heads-up"]
    engine._maybe_notify({**base, "bitcoin": {**b, "signal": "IN", "new_week": True}}, None, {})
    assert sent == ["Bitcoin heads-up", "Bitcoin trend: IN"]


def test_research_indicator_rules_match_the_live_signal():
    from tfsa_bot import research
    for tri in (make_tri({"btc": 0.003}), make_tri({})):
        assert research.indicator_rule("trend_score")(tri) == btc.is_in(tri)
    tri = make_tri({})
    tri.index = pd.bdate_range("2025-12-01", periods=len(tri))  # ends early 2027: ~33 months after the 2024 halving
    assert research.halving_calendar_rule(tri)
    assert not research.halving_calendar_rule(tri.loc[:"2026-06-30"])  # month 26: inside the OUT window


def test_research_btc_mix_in_and_out():
    from tfsa_bot import research
    core = pd.Series({"ca_eq": 0.25, "us_eq": 0.25, "btc": 0.15, "cash": 0.35})
    w = research.mix(core, True, 0.5)
    assert w["btc"] == pytest.approx(0.5 + 0.075)
    assert w["ca_eq"] == pytest.approx(0.125)
    assert w.sum() == pytest.approx(1.0)
    w = research.mix(core, False, 0.5)
    assert w["btc"] == pytest.approx(0.075)
    assert w["cash"] == pytest.approx(0.5 + 0.175)
    assert w.sum() == pytest.approx(1.0)


def test_research_btc_mix_band_avoids_small_weekly_trades():
    from tfsa_bot import research
    core = pd.Series({"ca_eq": 0.5, "cash": 0.5})
    assert research.mix(core, True, 0.5, current_btc=0.56)["btc"] == pytest.approx(0.56)  # drifted 6pts: keep
    assert research.mix(core, True, 0.5, current_btc=0.65)["btc"] == pytest.approx(0.5)   # 15pts: trim
    assert research.mix(core, True, 0.5, current_btc=0.0)["btc"] == pytest.approx(0.5)    # entering: full share
    assert research.mix(core, False, 0.5, current_btc=0.56)["btc"] == 0.0                 # OUT: sell it all
    for cur in (0.0, 0.4, 0.56, 0.9):
        w = research.mix(core, True, 0.5, current_btc=cur)
        assert w.sum() == pytest.approx(1.0) and (w >= -1e-12).all()


def test_research_core_view_sets_pot_aside():
    from tfsa_bot import research
    cur = pd.Series({"ca_eq": 0.3, "btc": 0.55, "cash": 0.15})
    assert research.core_view(cur, 0.5)["btc"] == pytest.approx(0.05)
    assert research.core_view(cur, 0.0)["btc"] == pytest.approx(0.55)
    assert research.core_view(None, 0.5) is None
