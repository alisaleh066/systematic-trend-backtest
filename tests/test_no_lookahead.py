import numpy as np
import pandas as pd

from stb.engine import build_fills, simulate
from stb.signals import momentum

T0 = 400


def perturbed(px):
    rng = np.random.default_rng(99)
    out = px.copy()
    out.iloc[T0 + 1 :] *= rng.uniform(0.5, 2.0, (len(px) - T0 - 1, px.shape[1]))
    return out


def test_fills_up_to_t0_ignore_later_prices(synth):
    px, rf = synth
    a = build_fills(px, 12, 0.10)
    b = build_fills(perturbed(px), 12, 0.10)
    early = [k for k in a if k <= T0 + 1]
    assert early, "need fills before the perturbation"
    assert sorted(k for k in b if k <= T0 + 1) == sorted(early)
    for k in early:
        assert np.array_equal(a[k], b[k])


def test_returns_up_to_t0_ignore_later_prices(synth):
    px, rf = synth
    ra = simulate(px, rf, build_fills(px, 12, 0.10), 10)
    pp = perturbed(px)
    rb = simulate(pp, rf, build_fills(pp, 12, 0.10), 10)
    assert np.array_equal(ra.net.iloc[: T0 + 1], rb.net.iloc[: T0 + 1])


def test_momentum_skips_the_last_month(synth):
    px, _ = synth
    p = px.to_numpy()
    i = 300
    base = momentum(p, i, 12)
    q = p.copy()
    q[i - 20 : i + 1] *= 3.0  # the 21 most recent sessions must not matter
    assert np.array_equal(base, momentum(q, i, 12))


def test_signal_day_return_is_unaffected_by_the_new_position(synth):
    px, rf = synth
    w = np.full(px.shape[1], 0.05)
    base = simulate(px, rf, {}, 10)
    filled = simulate(px, rf, {50: w}, 10)
    assert np.array_equal(base.net.iloc[:50], filled.net.iloc[:50])
    assert filled.net.iloc[50] < base.net.iloc[50]  # only the cost, no exposure yet
    assert filled.net.iloc[51] != base.net.iloc[51]  # exposure starts the day after the fill
