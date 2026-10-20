import json

import numpy as np
import pandas as pd
import pytest

from stb import seal2, universe
from stb.engine import simulate
from stb.engine_v2 import build_fills_v2, run_design, split_dev, target_weights_v2
from stb.signals_v2 import ensemble_signal

S1 = {"lookbacks_months": [12], "skip_sessions": 21}
S2 = {"lookbacks_months": [1, 3, 12], "skip_sessions": 0}
S3 = {"lookbacks_months": [3, 6, 12], "skip_sessions": 21}
LIST_DAY = 300


@pytest.fixture
def panel():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2012-01-02", periods=800)
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, (800, 6)), axis=0), idx, list("ABCDEF"))
    px.loc[idx[:LIST_DAY], "F"] = np.nan  # F lists late
    return px, pd.Series(0.0001, idx)


def test_ensemble_signal_values():
    t = np.arange(400, dtype=float)
    up = (100 + t)[:, None]
    down = (500 - t)[:, None]
    assert ensemble_signal(up, 399, **{"lookbacks_months": [1, 3, 12], "skip_sessions": 0})[0] == 1.0
    assert ensemble_signal(down, 399, **{"lookbacks_months": [3, 6, 12], "skip_sessions": 21})[0] == -1.0
    mixed = np.concatenate([np.linspace(100, 200, 380), np.linspace(200, 150, 20)])[:, None]  # last month down, year up
    assert ensemble_signal(mixed, 399, [1, 3, 12], 0)[0] == pytest.approx(-1 / 3)
    assert ensemble_signal(mixed, 399, [3, 6, 12], 21)[0] == 1.0  # the skip hides the last month


def test_asset_has_zero_weight_until_tradable(panel):
    px, _ = panel
    fills = build_fills_v2(px, list(px.columns), S1, 63, 0.10, 2.0, "2013-01-01", 273)
    col = list(px.columns).index("F")
    tradable_from = LIST_DAY + 272
    for k, w in fills.items():
        if k - 1 < tradable_from:
            assert w[col] == 0.0
    assert any(w[col] != 0 for k, w in fills.items() if k - 1 >= tradable_from)


def test_weights_ignore_future_prices(panel):
    px, _ = panel
    a = build_fills_v2(px, list(px.columns), S2, 63, 0.10, 2.0, "2013-01-01", 273)
    pp = px.copy()
    pp.iloc[601:] *= np.random.default_rng(1).uniform(0.5, 2.0, (len(px) - 601, 6))
    b = build_fills_v2(pp, list(px.columns), S2, 63, 0.10, 2.0, "2013-01-01", 273)
    early = [k for k in a if k <= 601]
    assert early and sorted(k for k in b if k <= 601) == sorted(early)
    assert all(np.array_equal(a[k], b[k]) for k in early)


def test_no_activity_before_window_start(panel):
    px, rf = panel
    res = run_design(px, rf, list(px.columns), S1, 63, 0.10, 10, "2014-06-02", 273)
    start = px.index.searchsorted(pd.Timestamp("2014-06-02"))
    assert (res.net.iloc[:start] == rf.iloc[:start].where(res.net.index[:start] >= res.net.index[1])).iloc[1:].all()
    assert res.turnover.iloc[: start + 1].sum() == 0 or res.turnover.iloc[:start].sum() == 0
    assert res.trades.empty or (res.trades["entry"] >= px.index[start]).all()


def test_cost_with_a_changing_universe():
    idx = pd.bdate_range("2020-01-01", periods=10)
    px = pd.DataFrame({"A": 100.0, "B": 100.0}, idx)
    rf = pd.Series(0.0, idx)
    fills = {3: np.array([0.4, 0.0]), 6: np.array([0.4, 0.3])}  # B joins at day 6
    res = simulate(px, rf, fills, 10, 0)
    assert res.cost.iloc[3] == pytest.approx(0.0004)
    assert res.cost.iloc[6] == pytest.approx(0.0003)  # only the newcomer is traded


def test_vol_target_is_hit_and_gross_is_capped():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, (300, 5))
    cov = np.cov(r, rowvar=False)
    w = target_weights_v2(np.array([1, 1, -1, 1, 1.0]), cov, 0.10, 2.0)
    assert np.sqrt(w @ cov @ w * 252) == pytest.approx(0.10, rel=1e-6)
    low = np.diag([1e-6] * 5)  # tiny vol would need huge leverage
    assert np.abs(target_weights_v2(np.ones(5), low, 0.10, 2.0)).sum() == pytest.approx(2.0)


def test_cash_rate_uses_previous_observation(monkeypatch):
    idx = pd.bdate_range("2020-01-01", periods=5)
    dtb = pd.Series([2.52, 5.04], pd.DatetimeIndex([idx[0], idx[3]]))
    monkeypatch.setattr(universe, "_read", lambda name: dtb)
    rf = universe.cash_returns(idx)
    assert rf.iloc[1] == pytest.approx(2.52 / 100 / 252)  # yield from t-1
    assert rf.iloc[3] == pytest.approx(2.52 / 100 / 252)  # new 5.04 only known from day 4
    assert rf.iloc[4] == pytest.approx(5.04 / 100 / 252)


def test_split_dev_is_chronological_and_month_end():
    idx = pd.bdate_range("2000-01-03", "2026-09-30")
    s = split_dev(idx, "2008-07-01", 0.75)
    assert pd.Timestamp(s["train"][1]) < pd.Timestamp(s["validation"][0])
    assert idx[idx > pd.Timestamp(s["train"][1])][0].month != pd.Timestamp(s["train"][1]).month
    assert abs(s["sessions"]["train"] / (s["sessions"]["train"] + s["sessions"]["validation"]) - 0.75) < 0.03


def test_seal2_refusals(tmp_path):
    p = tmp_path / "f.json"
    with pytest.raises(seal2.SealError, match="no freeze"):
        seal2.check_freeze2(clean=lambda: True, path=p)
    with pytest.raises(seal2.SealError, match="dirty"):
        seal2.write_freeze2({}, 49, {}, clean=lambda: False, path=p)
    seal2.write_freeze2({"x": 1}, 49, {}, clean=lambda: True, path=p)
    with pytest.raises(seal2.SealError, match="dirty"):
        seal2.check_freeze2(clean=lambda: False, path=p)
    for field in ("config_sha256", "code_sha256"):
        rec = json.loads(p.read_text())
        rec[field] = "0" * 64
        q = tmp_path / f"{field}.json"
        q.write_text(json.dumps(rec))
        with pytest.raises(seal2.SealError, match="changed since the freeze"):
            seal2.check_freeze2(clean=lambda: True, path=q)
