import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from stb.radar import benjamini_hochberg, break_vs_hold, classify_rules

FD = Path(r"C:\Users\ali12\Financedashboard")


def vectors():
    rng = np.random.default_rng(11)
    out = []
    for seed in range(25):
        r = np.random.default_rng(seed)
        x = 100 * np.cumprod(1 + r.normal(0, 0.01, 400))
        y = 50 * np.cumprod(1 + 0.6 * np.diff(np.concatenate([[0], np.log(x)])) + r.normal(0, 0.01, 400))
        if seed % 5 == 0:
            x[-1] *= 1.2  # force a 20-session extreme in one leg
        out.append((list(x), list(y)))
    return out


def load_original():
    """Execute the literal source of correlation/changes/classify from the original file (its imports need httpx etc.)."""
    import ast
    import math
    import statistics
    import types

    src = (FD / "server" / "routine_radar.py").read_text(encoding="utf-8")
    wanted = {"correlation", "changes", "classify"}
    nodes = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    mod = types.SimpleNamespace()
    ns = {"statistics": statistics, "finite": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}
    exec(compile(ast.Module(nodes, []), "routine_radar.py", "exec"), ns)  # noqa: S102 - trusted local file
    mod.classify = ns["classify"]
    return mod


def test_matches_original_classify():
    if not (FD / "server" / "routine_radar.py").exists():
        pytest.skip("Financedashboard not on this machine")
    mod = load_original()
    for x, y in vectors():
        for expected, rc in ((1, False), (-1, True)):
            ref = mod.classify(x, y, expected, False, False, rc)
            ours = classify_rules(x, y, expected, False, False, rc)
            if ref["status"] == "no read":
                assert ours is None
            else:
                assert ours is not None
                assert ours["any_break"] == (ref["status"] == "breaks")
                flags = " ".join(ref["flags"])
                assert ours["correlation"] == ("60-session correlation" in flags)
                assert ours["zscore"] == ("ratio z-score" in flags)


def test_short_history_is_no_read():
    assert classify_rules([1.0] * 100, [1.0] * 100) is None


def test_break_vs_hold_detects_a_planted_difference():
    rng = np.random.default_rng(0)
    month = np.repeat(np.arange(60), 5)
    brk = rng.random(300) < 0.3
    y = rng.normal(0, 0.02, 300) - 0.05 * brk
    res = break_vs_hold(month, brk, y, resamples=2000)
    assert res["diff"] < -0.03 and res["ci95"][1] < 0


def test_bh_controls_the_rank_order():
    assert benjamini_hochberg([0.001, 0.2, 0.9], 0.05) == [True, False, False]
