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


def _note_summary(**bitcoin) -> dict:
    b = {"signal": "OUT", "previous_week": "OUT", "new_week": False, "latest": {"state": "DOWN"}, **bitcoin}
    return {"signal_week": "2026-11-13", "changes": [], "new_week": False, "news": None, "news_fresh": False,
            "scan": None, "bitcoin": b}


def test_bitcoin_notification_states_map_to_in_out(monkeypatch):
    from tfsa_bot import engine
    sent = []
    monkeypatch.setattr(engine.notify, "send", lambda t, b: sent.append(t))
    up = _note_summary(latest={"state": "UP"})
    engine._maybe_notify(up, None)
    assert sent == ["Bitcoin heads-up"]
    engine._maybe_notify(up, None)             # already told
    engine._maybe_notify(_note_summary(), None)  # agrees again: forget the heads-up
    assert sent == ["Bitcoin heads-up"]
    engine._maybe_notify(up, None)             # disagrees again later: tell again
    assert sent == ["Bitcoin heads-up"] * 2
    engine._maybe_notify({**_note_summary(signal="IN", latest={"state": "UP"}), "signal_week": "2026-11-20"}, None)
    assert sent == ["Bitcoin heads-up"] * 2 + ["Bitcoin trend: IN"]


def test_report_opened_without_notifying_does_not_swallow_notifications(monkeypatch):
    """The app and the CLI log runs too (send=False). They used to count as "already told"."""
    from tfsa_bot import engine
    sent = []
    monkeypatch.setattr(engine.notify, "send", lambda t, b: sent.append((t, b)))
    ev = {"buy": [{"ticker": "XIT.TO"}], "sell": [{"ticker": f"S{i}.TO"} for i in range(8)]}
    s = {**_note_summary(latest={"state": "UP"}), "scan": {"events": ev},
         "changes": [{"action": "BUY", "ticker": "XIT.TO", "name": "Canadian tech", "from": 0.0, "to": 0.25}]}
    db.save_run({**s, "new_week": True})  # what get_brief does right after Friday's close
    engine._maybe_notify(s, None)
    assert [t for t, _ in sent] == ["New weekly signal", "Trend flips this week", "Bitcoin heads-up"]
    assert sent[1][1] == "BUY: XIT.TO; SELL: S0.TO, S1.TO, S2.TO, S3.TO, S4.TO, S5.TO +2 more"
    monkeypatch.setattr(engine, "datetime", type("D", (), {"now": staticmethod(lambda tz=None: pd.Timestamp("2026-11-18 12:30"))}))
    engine._maybe_notify(s, None)  # same week, a Wednesday: nothing new
    assert len(sent) == 3


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


# ---- per-asset track record, events, scan ---------------------------------------------------------

def wavy(days: int = 1500, seed: int = 1) -> pd.DataFrame:
    """Daily returns for a Bitcoin-like asset that trends up and down in turns, plus cash."""
    idx = pd.bdate_range("2015-01-01", periods=days)
    rng = np.random.default_rng(seed)
    drift = 0.002 * np.sin(np.arange(days) / 60)
    return pd.DataFrame({"btc": drift + rng.normal(0, 0.01, days), "cash": 0.0001}, index=idx)


def test_track_record_matches_the_backtest_engine():
    from tfsa_bot import backtest as bt
    from tfsa_bot import research
    rets = wavy()
    tri = (1 + rets).cumprod()
    rec = signals.track_record(tri, "btc")
    ref = bt.run(rets, research.btc_rule_fn(research.indicator_rule("trend_score")), start=rec["from"]).stats
    assert rec["flips_per_year"] > 1  # the series really switches
    assert rec["signal_cagr"] == pytest.approx(ref["CAGR"], abs=1e-4)
    assert rec["signal_max_dd"] == pytest.approx(ref["MaxDD"], abs=1e-4)
    assert rec["flips_per_year"] == pytest.approx(ref["Trades/yr"], abs=0.05)
    assert rec["time_in"] == pytest.approx(1 - ref["Cash%"], abs=0.01)
    hold = (1 + rets["btc"].loc[rec["from"]:]).cumprod()
    assert rec["hold_max_dd"] == pytest.approx(float((hold / hold.cummax() - 1).min()), abs=1e-4)
    assert signals.track_record(tri.iloc[:400], "btc") is None  # under a year of signals


def test_flags_buy_sell_pending_near():
    up = {"state": "UP", "flipped": True}
    assert signals.flags(up, {"state": "UP", "flip_move": -0.2}, 0.03) == {"event": "BUY", "pending": None, "near_flip": False}
    assert signals.flags({"state": "DOWN", "flipped": True}, {"state": "DOWN", "flip_move": 0.01}, 0.03) == \
        {"event": "SELL", "pending": None, "near_flip": True}
    held = {"state": "UP", "flipped": False}
    assert signals.flags(held, {"state": "DOWN", "flip_move": 0.01}, 0.03) == {"event": None, "pending": "DOWN", "near_flip": False}
    assert signals.flags({"state": None}, {"state": None, "flip_move": None}, None) == {"event": None, "pending": None, "near_flip": False}


def test_asset_signal_marks_a_flip_only_when_the_week_before_disagreed():
    idx = pd.bdate_range("2023-01-02", periods=400)
    lvl = pd.Series(np.linspace(100, 80, 400), index=idx)
    lvl.iloc[-2:] = 140.0  # jumps in the final week: last week-end UP, the one before DOWN
    tri = pd.DataFrame({"btc": lvl, "cash": 1.0})
    a = signals.asset_signal(tri, tri, "btc")
    assert a["weekly"]["state"] == "UP" and a["weekly"]["weeks"] == 1 and a["weekly"]["flipped"]
    assert a["event"] == "BUY"
    first = signals.asset_signal(tri.iloc[:254], tri.iloc[:254], "btc")  # its first signal ever is not a flip
    assert first["weekly"]["weeks"] == 1 and not first["weekly"]["flipped"] and first["event"] is None


def test_model_status_explains_the_hold_buffer():
    from tfsa_bot.engine import model_status
    p = Params()
    assert model_status(2, "UP", True, p) == "held"
    assert model_status(6, "UP", True, p) == "held_by_buffer"
    assert model_status(4, "UP", False, p) == "next_in_line"
    assert model_status(5, "UP", False, p) is None and model_status(3, "DOWN", False, p) is None


def test_stale_prices_are_flagged_and_lose_their_signal():
    from tfsa_bot import data
    cal = pd.bdate_range("2026-01-01", periods=30)
    px = pd.DataFrame({"OK": 1.0, "DAY": 1.0, "OLD": 1.0, "GONE": np.nan}, index=cal)
    px.loc[cal[-1]:, "DAY"] = np.nan  # one day behind: normal (US vs Canadian holidays)
    px.loc[cal[-7]:, "OLD"] = np.nan
    assert data.stale(px, cal, cal[-1]) == {
        "OLD": {"last": cal[-8].date().isoformat(), "trading_days_old": 7},
        "GONE": {"last": None, "trading_days_old": None},
    }
    lvl = data.series_index(px["OLD"].dropna(), cal)
    assert lvl.iloc[-7:-2].notna().all() and lvl.iloc[-2:].isna().all()  # filled for 5 days, then no price
    drag = data.series_index(pd.Series(1.0, index=cal), cal, drag=0.0252)
    assert drag.iloc[-1] == pytest.approx((1 - 0.0001) ** 29)


def test_watchlist_is_separate_from_the_strategy_universe():
    from tfsa_bot.universe import SLEEVES, WATCHLIST
    tickers = [w.ticker for w in WATCHLIST]
    assert len(tickers) == len(set(tickers))
    assert not set(tickers) & {s.ticker for s in SLEEVES}
    assert all(w.source and w.drag > 0 for w in WATCHLIST if w.kind == "cdr")
    assert all(w.kind in __import__("tfsa_bot.scan", fromlist=["KINDS"]).KINDS for w in WATCHLIST)


def test_edit_watchlist():
    from tfsa_bot import scan
    cfg = scan.edit_watchlist({"follow": "core"}, add=["bce.to", "CHPS.TO"], remove=["XGD.TO"])
    assert cfg == {"follow": "core", "watchlist_extra": ["BCE.TO"], "watchlist_hidden": ["XGD.TO"]}
    names = [w.ticker for w in scan.watchlist(cfg)]
    assert "BCE.TO" in names and "XGD.TO" not in names and "CHPS.TO" in names
    cfg = scan.edit_watchlist(cfg, add=["XGD.TO"], remove=["BCE.TO"])
    assert cfg["watchlist_extra"] == [] and cfg["watchlist_hidden"] == []
    assert scan.find_watch("nvda", cfg).ticker == "NVDA.NE" and scan.find_watch("BRK-B", cfg).ticker == "BRK.NE"
    assert scan.find_watch("shop", cfg).ticker == "SHOP.TO" and scan.find_watch("ZZZZ", cfg) is None


def test_scan_covers_both_tiers_and_never_touches_the_model(monkeypatch):
    from tfsa_bot import data, engine, report, scan
    from tfsa_bot.universe import BY_KEY
    tri = make_tri({"ca_banks": 0.003, "ca_energy": 0.0025, "gold": 0.002, "us_eq": 0.0015, "bonds": -0.001}, days=600)
    raw = tri.drop(columns="cash").rename(columns=lambda k: BY_KEY[k].ticker)
    m = engine.Market(raw, tri.pct_change().fillna(0.0), tri, tri.index[-1])
    idx = tri.index
    watch = pd.DataFrame({
        "CHPS.TO": np.linspace(10, 30, 600),                      # steady uptrend
        "ZID.TO": np.linspace(30, 10, 600),                       # steady downtrend
        "NVDA": np.linspace(100, 400, 600),                       # the CDR's history
        "SOLQ.TO": [np.nan] * 500 + list(np.linspace(5, 9, 100)),  # under 12 months
    }, index=idx)
    watch.loc[idx[-4]:, "ZID.TO"] = np.nan  # Yahoo stopped updating it
    monkeypatch.setattr(data, "load_watch", lambda tickers, refresh=False: watch.reindex(columns=sorted(set(tickers))))
    before = target_weights(m.tri_w)

    sc = scan.build(m, {}, held={"ca_banks"})
    rows = {r["ticker"]: r for r in sc["rows"]}
    assert len(rows) == len(universe_keys(Params())) + len(scan.watchlist({}))
    assert rows["ZEB.TO"]["tier"] == "model" and rows["ZEB.TO"]["in_model"] and rows["ZEB.TO"]["record"]
    assert rows["CHPS.TO"]["tier"] == "watch" and rows["CHPS.TO"]["trend"] == "UP" and not rows["CHPS.TO"]["in_model"]
    assert rows["NVDA.NE"]["trend"] == "UP" and rows["NVDA.NE"]["priced_as"] == "NVDA"
    assert rows["SOLQ.TO"]["trend"] is None and rows["SOLQ.TO"]["stale"] is None  # too young, not stale
    assert rows["ZID.TO"]["stale"]["trading_days_old"] == 4 and "ZID.TO" in sc["stale"]
    assert rows["XGD.TO"]["trend"] is None and rows["XGD.TO"]["stale"] == {"last": None, "trading_days_old": None}
    assert all(i["ticker"] != "ZID.TO" for group in sc["events"].values() for i in group)
    scores_ = [r["score"] for r in sc["rows"] if r["score"] is not None]
    assert scores_ == sorted(scores_, reverse=True)
    assert sc["counts"]["up"] + sc["counts"]["down"] + sc["counts"]["no_signal"] == len(rows)
    assert {r["ticker"] for r in scan.select(sc["rows"], kind="cdr")} == {w.ticker for w in scan.watchlist({}) if w.kind == "cdr"}
    assert all(r["trend"] == "UP" for r in scan.select(sc["rows"], show="up"))
    pd.testing.assert_series_equal(before, target_weights(m.tri_w))  # the scan is information only

    text = report.scan_text(sc)
    assert "CHPS.TO" in text and "STALE" in text and "needs 12 months of history" in text
    a = engine.asset_report(m, "nvda")
    assert a["ticker"] == "NVDA.NE" and a["price_ticker"] == "NVDA" and "CDR" in a["note"]
    assert a["record"] and "Track record of following this signal" in report.signal_text(a)


# ---- on-chain -------------------------------------------------------------------------------------

def chain_raw(days: int = 1500, seed: int = 2) -> pd.DataFrame:
    """Synthetic CoinMetrics-like daily history (UTC dates)."""
    from tfsa_bot import onchain
    idx = pd.date_range("2016-01-01", periods=days, freq="D")
    rng = np.random.default_rng(seed)
    price = 1000 * np.exp(np.cumsum(0.002 * np.sin(np.arange(days) / 90) + rng.normal(0, 0.02, days)))
    mvrv = 1.5 + np.sin(np.arange(days) / 120)
    return pd.DataFrame({
        "PriceUSD": price, "CapMrktCurUSD": price * 1.9e7, "CapMVRVCur": mvrv,
        "IssTotUSD": price * 900 * (1 + 0.1 * rng.normal(size=days)), "HashRate": np.linspace(1, 5, days),
        "AdrActCnt": 8e5 + rng.normal(0, 1e4, days),
    }, index=idx)[list(onchain.METRICS)]


def test_onchain_features_are_point_in_time():
    from tfsa_bot import onchain
    raw = chain_raw()
    full, cut = onchain.features(raw), onchain.features(raw.iloc[:1000])
    cols = ["mvrv_pct", "puell_pct", "mvrv_z", "puell", "hash_ribbon", "active_addr_trend"]
    pd.testing.assert_frame_equal(full[cols].iloc[:1000], cut[cols])  # future data never changes the past
    f = full.dropna(subset=["mvrv_pct"])
    assert f["mvrv_pct"].between(0, 1).all()
    assert full["realized_price_usd"].iloc[-1] == pytest.approx(raw["PriceUSD"].iloc[-1] / raw["CapMVRVCur"].iloc[-1])


def test_onchain_value_is_used_the_day_after_its_utc_date():
    from tfsa_bot import onchain
    feat = onchain.features(chain_raw())
    d = pd.Timestamp("2019-06-12")  # a Wednesday
    cal = onchain.on_calendar(feat, pd.DatetimeIndex([d]))
    assert cal.loc[d, "mvrv"] == feat.loc[d - pd.Timedelta(days=1), "mvrv"]
    assert onchain.known_at(feat, d)["mvrv"] == cal.loc[d, "mvrv"]  # live and backtest see the same row
    assert onchain.known_at(feat, feat.index[-1] + pd.Timedelta(days=10)) is None  # stale: no reading


def test_onchain_panel_and_zones():
    from tfsa_bot import onchain
    feat = onchain.features(chain_raw())
    o = onchain.panel(feat, feat.index[-1] + pd.Timedelta(days=1))
    assert o["data_date"] == feat.index[-1].date().isoformat()
    assert o["nupl"] == pytest.approx(1 - 1 / o["mvrv"], abs=1e-3)
    assert o["mvrv_zone"] == onchain.zone(o["mvrv_pct"])
    assert onchain.zone(0.95) == "high" and onchain.zone(0.05) == "low" and onchain.zone(0.5) == "middle"
    assert "error" in onchain.panel(feat, feat.index[-1] + pd.Timedelta(days=30))


def test_onchain_early_entry_only_adds_near_lows_and_research_uses_the_same_rule():
    from tfsa_bot import onchain, research
    assert onchain.early_entry(True, 0.99) and onchain.early_entry(True, None)  # never leaves earlier than the signal
    assert onchain.early_entry(False, 0.05) and not onchain.early_entry(False, 0.5)
    assert not onchain.early_entry(False, None) and not onchain.early_entry(False, float("nan"))
    tri = make_tri({})  # flat: the trend signal is OUT
    feat = pd.DataFrame({"mvrv_pct": 0.05, "puell_pct": 0.5, "hash_ribbon": 0.01}, index=tri.index)
    rules = research.onchain_rules(feat)
    assert rules["SHADOW: signal OR MVRV <= 10th pct (early entry)"](tri) == 1.0
    assert rules["SIGNAL: trend score"](tri) == 0.0
    up = make_tri({"btc": 0.003})
    hot = pd.DataFrame({"mvrv_pct": 0.95, "puell_pct": 0.95, "hash_ribbon": -0.01}, index=up.index)
    for name, rule in research.onchain_rules(hot).items():
        if name.startswith("signal, "):  # the cut variants only ever reduce the signal's position
            assert 0.0 <= rule(up) < research.onchain_rules(hot)["SIGNAL: trend score"](up) == 1.0


def test_bitcoin_report_survives_missing_onchain_data(monkeypatch):
    from tfsa_bot import engine
    rets = wavy()
    tri = (1 + rets).cumprod()
    m = engine.Market(pd.DataFrame(index=rets.index), rets, tri, tri.index[-6])
    out = engine.bitcoin_report(m, Params(), chain="URLError: offline")
    assert out["onchain"] == {"error": "URLError: offline"}
    assert out["signal"] in ("IN", "OUT")


def test_shadow_onchain_track_records_early_entry_and_skips_without_data():
    from tfsa_bot import engine, onchain
    tri = make_tri({})  # flat Bitcoin: the trend signal is OUT
    rets = tri.pct_change().fillna(0)
    m = engine.Market(pd.DataFrame(index=tri.index), rets, tri, tri.index[-1])
    engine._record_paper_tracks(m, Params(), None, None, "normal", {}, "offline")
    assert db.decisions("btc")[-1]["weights"] == {"cash": 1.0}
    assert db.decisions("btc_onchain") == []  # no data: nothing guessed
    days = pd.date_range(tri.index[0] - pd.Timedelta(days=5), tri.index[-1], freq="D")
    feat = pd.DataFrame({"mvrv": 0.8, "mvrv_pct": 0.04}, index=days)
    engine._record_paper_tracks(m, Params(), None, None, "normal", {}, feat)
    d = db.decisions("btc_onchain")[-1]
    assert d["weights"] == {"btc": 1.0} and d["notes"]
    assert onchain.known_at(feat, m.week)["mvrv_pct"] == 0.04


# ---- resilience, blended ranking, quality metrics & opportunities ---------------------------------

def test_load_raw_and_watch_preserve_cache_on_failed_or_partial_download(tmp_path, monkeypatch):
    from tfsa_bot import data
    idx = pd.bdate_range("2024-01-01", periods=30)
    good = pd.DataFrame({"XIC.TO": np.linspace(30, 35, 30), "VFV.TO": np.linspace(100, 120, 30)}, index=idx)
    raw_p = tmp_path / "raw.pkl"
    watch_p = tmp_path / "watch.pkl"
    monkeypatch.setattr(data, "RAW_CACHE", raw_p)
    monkeypatch.setattr(data, "WATCH_CACHE", watch_p)
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data, "_all_tickers", lambda: ["VFV.TO", "XIC.TO"])
    data._write_pickle(good, raw_p)
    data._write_pickle(good, watch_p)

    # Empty download must NOT overwrite valid cache
    monkeypatch.setattr(data, "_download_closes", lambda tickers: pd.DataFrame(columns=tickers))
    got = data.load_raw(refresh=True)
    pd.testing.assert_frame_equal(got, good)
    pd.testing.assert_frame_equal(pd.read_pickle(raw_p), good)
    pd.testing.assert_frame_equal(data.load_watch(["XIC.TO", "VFV.TO"], refresh=True), good[["VFV.TO", "XIC.TO"]])

    # Exception must fall back to cache
    def boom(tickers):
        raise ConnectionError("offline")
    monkeypatch.setattr(data, "_download_closes", boom)
    pd.testing.assert_frame_equal(data.load_raw(refresh=True), good)

    # Partial outage (VFV.TO all NaN) merges cached VFV.TO instead of wiping it
    partial = pd.DataFrame({"XIC.TO": np.linspace(30, 36, 30), "VFV.TO": np.nan}, index=idx)
    monkeypatch.setattr(data, "_download_closes", lambda tickers: partial)
    merged = data.load_raw(refresh=True)
    assert merged["VFV.TO"].notna().all() and merged["XIC.TO"].iloc[-1] == pytest.approx(36.0)


def test_blended_ranking_prefers_smoother_uptrend():
    from tfsa_bot.strategy import QUALITY_PARAMS
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(7)
    rets = pd.DataFrame(0.0, index=idx, columns=KEYS)
    rets["cash"] = 0.0001
    # Smooth uptrend vs slightly higher raw return but 10x higher noise
    rets["gold"] = 0.0012 + rng.normal(0, 0.001, 300)
    rets["ca_energy"] = 0.0014 + rng.normal(0, 0.03, 300)
    tri = (1 + rets).cumprod()
    p_raw = Params(top_n=1, hold_buffer=0, ranking="raw")
    p_blend = Params(top_n=1, hold_buffer=0, ranking="blend", vol_window=126)
    w_raw = target_weights(tri, p_raw)
    w_blend = target_weights(tri, p_blend)
    assert w_blend["gold"] == pytest.approx(1.0)  # smoother trend wins blended rank
    assert QUALITY_PARAMS.ranking == "blend" and QUALITY_PARAMS.vol_window == 126


def test_signal_quality_metrics_rsi_vol_flip_weeks_and_empty_guard():
    tri = make_tri({"gold": 0.002})
    a = signals.asset_signal(tri, tri, "gold", close=40.0)
    assert a["ann_vol"] is not None and a["ann_vol"] > 0
    assert a["risk_adj_score"] is not None and a["risk_adj_score"] > 0
    assert a["rsi_14"] is not None and 50.0 < a["rsi_14"] <= 100.0
    assert a["flip_weeks"] is not None and a["flip_weeks"] > 0
    # Empty series must not crash
    empty_tri = tri.copy()
    empty_tri["gold"] = np.nan
    e = signals.asset_signal(empty_tri, empty_tri, "gold")
    assert e["weekly"]["state"] is None and e["rsi_14"] is None


def test_opportunities_compare_and_shadow_quality_track(monkeypatch):
    from tfsa_bot import data, engine, report, scan
    from tfsa_bot.universe import BY_KEY
    tri = make_tri({"ca_banks": 0.003, "ca_energy": 0.0025, "gold": 0.002, "us_eq": 0.0015, "bonds": -0.001}, days=400)
    raw = tri.drop(columns="cash").rename(columns=lambda k: BY_KEY[k].ticker)
    m = engine.Market(raw, tri.pct_change().fillna(0.0), tri, tri.index[-1])
    monkeypatch.setattr(data, "load_watch", lambda tickers, refresh=False: pd.DataFrame(index=tri.index, columns=sorted(set(tickers))))

    sc = scan.build(m, {}, held={"ca_banks"})
    by_q = scan.select(sc["rows"], tier="model", sort="risk_adj")
    assert all(r["tier"] == "model" for r in by_q)
    qs = [r["risk_adj_score"] for r in by_q if r.get("risk_adj_score") is not None]
    assert qs == sorted(qs, reverse=True)

    opp = engine.opportunities_report(m, tier="model")
    assert "strongest_uptrends" in opp and "model" in opp
    txt = report.opportunities_text(opp)
    assert "Opportunities —" in txt and "MODEL PORTFOLIO" in txt

    comp = engine.compare_report(m, ["ZEB", "CGL-C.TO", "ZAG"])
    assert len(comp["assets"]) == 3 and not comp["errors"]
    assert "Comparison —" in report.compare_text(comp)

    # Shadow quality track is recorded alongside core/overlay
    engine._record_paper_tracks(m, Params(), None, None, "normal", {}, "offline", None)
    assert len(db.decisions("quality")) == 1
    assert sum(db.decisions("quality")[0]["weights"].values()) == pytest.approx(1.0, abs=1e-3)
