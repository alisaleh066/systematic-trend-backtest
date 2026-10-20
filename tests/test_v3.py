import json

import numpy as np
import pandas as pd
import pytest

from stb import seal3
from stb.engine import simulate
from stb.engine_v2 import build_fills_v2, run_design
from stb.universe_v3 import excess_return_index, fit_to_calendar, fx_usd_per_unit

S1 = {"lookbacks_months": [12], "skip_sessions": 21}


def test_fx_inversion():
    raw = pd.Series([100.0, 125.0])  # JPY per USD
    assert list(fx_usd_per_unit(raw, "invert")) == [0.01, 0.008]
    assert list(fx_usd_per_unit(raw, "direct")) == [100.0, 125.0]


def test_excess_proxy_earns_exactly_the_spot_move():
    idx = pd.bdate_range("2020-01-01", periods=6)
    spot = pd.Series([np.nan, 1.0, 1.1, 1.1, 0.99, 1.0], idx)
    rf = pd.Series(0.0002, idx)
    px = excess_return_index(spot, rf).to_frame("FX")
    res = simulate(px.bfill(), rf, {1: np.array([1.0])}, 0, 0)  # fully long from day 1, no costs
    excess = (res.net - rf).iloc[2:]
    assert np.allclose(excess, spot.pct_change().iloc[2:])


def test_gap_rule():
    cal = pd.bdate_range("2020-01-01", periods=12)
    s = pd.Series(1.0, cal)
    two = s.drop(cal[[4, 5]])
    filled, why = fit_to_calendar(two, cal, 2)
    assert why is None and filled.notna().all()
    three = s.drop(cal[[4, 5, 6]])
    assert fit_to_calendar(three, cal, 2) == (None, "gap of 3 sessions in a row (limit 2)")
    stale = s.iloc[:3]
    assert fit_to_calendar(stale, cal, 2)[0] is None


def mixed_panel():
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("1990-01-01", periods=900)
    rf = pd.Series(0.0002, idx)
    funds = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, (900, 3)), axis=0), idx, ["F1", "F2", "F3"])
    spot = pd.Series(np.cumprod(1 + rng.normal(0, 0.006, 900)), idx)
    spot.iloc[:350] = np.nan
    funds["FX"] = excess_return_index(spot, rf)
    return funds, rf


def test_mixed_panel_no_lookahead_and_listing():
    px, rf = mixed_panel()
    a = build_fills_v2(px, list(px.columns), S1, 126, 0.10, 2.0, "1991-03-01", 273)
    pp = px.copy()
    pp.iloc[701:] *= 1.7
    b = build_fills_v2(pp, list(px.columns), S1, 126, 0.10, 2.0, "1991-03-01", 273)
    early = [k for k in a if k <= 701]
    assert early and all(np.array_equal(a[k], b[k]) for k in early)
    fx = list(px.columns).index("FX")
    assert all(w[fx] == 0 for k, w in a.items() if k - 1 < 350 + 272)


def test_no_trading_before_window_start():
    px, rf = mixed_panel()
    res = run_design(px, rf, list(px.columns), S1, 126, 0.10, 10, "1992-06-01", 273)
    start = px.index.searchsorted(pd.Timestamp("1992-06-01"))
    assert res.turnover.iloc[:start].sum() == 0
    assert res.trades.empty or (res.trades["entry"] >= px.index[start]).all()


def test_seal3_refusals(tmp_path):
    p = tmp_path / "f.json"
    with pytest.raises(seal3.SealError, match="no freeze"):
        seal3.check_freeze3(clean=lambda: True, path=p)
    with pytest.raises(seal3.SealError, match="dirty"):
        seal3.write_freeze3(67, {}, clean=lambda: False, path=p)
    seal3.write_freeze3(67, {}, clean=lambda: True, path=p)
    with pytest.raises(seal3.SealError, match="dirty"):
        seal3.check_freeze3(clean=lambda: False, path=p)
    for field in ("config_sha256", "code_sha256"):
        rec = json.loads(p.read_text())
        rec[field] = "0" * 64
        q = tmp_path / f"{field}.json"
        q.write_text(json.dumps(rec))
        with pytest.raises(seal3.SealError, match="changed since the freeze"):
            seal3.check_freeze3(clean=lambda: True, path=q)
