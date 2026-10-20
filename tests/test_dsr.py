import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from stb.dsr import deflated_sharpe

REF = Path(r"C:\Users\ali12\Financedashboard\server\analysis_engine.py")


def sample():
    rng = np.random.default_rng(3)
    return rng.normal(0.0006, 0.01, 800), list(rng.normal(0.02, 0.03, 20))


def test_more_trials_lowers_the_probability():
    x, trials = sample()
    assert deflated_sharpe(x, trials, 50)["probability"] < deflated_sharpe(x, trials, 5)["probability"]


def test_matches_financedashboard_reference():
    if not REF.exists():
        pytest.skip("Financedashboard not on this machine")
    sys.path.insert(0, str(REF.parents[1]))
    try:
        spec = importlib.util.spec_from_file_location("fd_analysis_engine", REF)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"reference not importable: {exc}")
    finally:
        sys.path.pop(0)
    x, trials = sample()
    ref = mod.deflated_sharpe(list(x), trials, 20)
    ours = deflated_sharpe(x, trials, 20)
    assert ours["probability"] == pytest.approx(ref["score"], rel=1e-9)


def test_degenerate_inputs_return_none():
    x, trials = sample()
    assert deflated_sharpe(x[:10], trials, 20) is None
    assert deflated_sharpe(x, [0.01, 0.01], 20) is None
