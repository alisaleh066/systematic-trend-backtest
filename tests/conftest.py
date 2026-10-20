import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synth():
    """16 random-walk assets plus a constant cash rate, 700 business days."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015-01-01", periods=700)
    rets = rng.normal(0.0003, 0.01, (700, 16))
    px = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), idx, [f"A{i}" for i in range(16)])
    rf = pd.Series(0.0001, idx)
    return px, rf


def flat_prices(n_days=20, n_assets=1):
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(100.0, idx, [f"A{i}" for i in range(n_assets)]), pd.Series(0.0, idx)
