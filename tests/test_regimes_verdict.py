import numpy as np
import pandas as pd

from stb.regimes import efficiency_ratio, regime_labels
from stb.verdict import decide, neighbour_gap


def test_efficiency_ratio_trend_vs_chop():
    idx = pd.bdate_range("2020-01-01", periods=200)
    up = pd.Series(np.linspace(100, 200, 200), idx)
    chop = pd.Series(100 + np.tile([0.0, 1.0], 100), idx)
    assert efficiency_ratio(up).iloc[-1] == 1.0
    assert efficiency_ratio(chop).iloc[-1] < 0.1
    assert np.isnan(efficiency_ratio(pd.Series(100.0, idx)).iloc[-1])  # zero denominator is flagged, not inf


def test_labels_use_only_earlier_data():
    idx = pd.bdate_range("2018-01-01", periods=600)
    rng = np.random.default_rng(1)
    c = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 600)), idx)
    a = regime_labels(c)
    c2 = c.copy()
    c2.iloc[400:] *= 5
    b = regime_labels(c2)
    assert (a["cell"].iloc[:401] == b["cell"].iloc[:401]).all()  # label at t ignores t's own close
    assert (a["cell"].iloc[:300] == "unknown").sum() >= 250  # warm-up is unknown


def test_neighbour_gap_and_verdict():
    lbs, vols = [3, 6, 9, 12], [0.06, 0.08, 0.10, 0.12, 0.15]
    cells = {(l, v): 0.2 for l in lbs for v in vols}
    cells[(12, 0.10)] = 1.0
    assert abs(neighbour_gap(cells, (12, 0.10), lbs, vols) - 0.8) < 1e-9
    good = {"total_return": 0.3, "profit_factor": 1.5, "round_trips": 150}
    assert decide(good, good, 0.1, 2, [], 100)["verdict"] == "PROMOTE"
    assert decide(good, good, 0.1, 2, ["range/high"], 100)["verdict"] == "WATCH"
    assert decide({**good, "total_return": -0.1}, good, 0.1, 2, [], 100)["verdict"] == "REJECT"
