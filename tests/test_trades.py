import numpy as np
import pandas as pd
import pytest

from stb.engine import simulate


def test_round_trip_counting():
    idx = pd.bdate_range("2020-01-01", periods=14)
    px = pd.DataFrame({"A": np.linspace(100, 113, 14)}, idx)
    rf = pd.Series(0.0, idx)
    fills = {2: [1.0], 4: [0.5], 6: [-1.0], 8: [0.0], 10: [1.0]}  # open, resize, reverse, close, open (never closed)
    res = simulate(px, rf, {k: np.array(v) for k, v in fills.items()}, 10)
    t = res.trades
    assert len(t) == 2  # long 2->6 and short 6->8; resizing and the trailing open trade do not count
    assert list(t["sign"]) == [1, -1]
    assert t["entry"].iloc[0] == idx[2] and t["exit"].iloc[0] == idx[6]
    assert t["gross_ret"].iloc[0] == pytest.approx(px["A"].iloc[6] / px["A"].iloc[2] - 1)
    assert t["net_ret"].iloc[0] == pytest.approx(t["gross_ret"].iloc[0] - 2 * 10 / 1e4)
