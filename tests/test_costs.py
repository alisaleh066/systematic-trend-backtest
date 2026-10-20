import numpy as np
import pytest

from stb.engine import simulate
from tests.conftest import flat_prices

TARGET = np.array([0.5, -0.5])


@pytest.mark.parametrize("bps", [10, 20, 30])
def test_cost_is_bps_times_turnover(bps):
    px, rf = flat_prices(12, 2)
    res = simulate(px, rf, {3: TARGET}, bps, borrow_bps=0)
    assert res.turnover.iloc[3] == pytest.approx(1.0)
    assert res.cost.iloc[3] == pytest.approx(bps / 1e4)
    assert res.net.iloc[3] == pytest.approx(-bps / 1e4)


def test_no_trades_means_no_cost():
    px, rf = flat_prices(12, 2)
    res = simulate(px, rf, {}, 30)
    assert res.cost.sum() == 0 and res.net.abs().sum() == 0


def test_refill_of_unchanged_target_costs_nothing():
    px, rf = flat_prices(12, 2)
    res = simulate(px, rf, {3: TARGET, 6: TARGET}, 10, borrow_bps=0)
    assert res.cost.iloc[6] == pytest.approx(0.0)


def test_borrow_charged_on_short_notional_only():
    px, rf = flat_prices(12, 2)
    res = simulate(px, rf, {3: TARGET}, 10, borrow_bps=25)
    assert res.borrow.iloc[4] == pytest.approx(0.0025 / 252 * 0.5)
    long_only = simulate(px, rf, {3: np.array([0.5, 0.5])}, 10, borrow_bps=25)
    assert long_only.borrow.sum() == 0


def test_stress_doubles_and_triples_cost():
    px, rf = flat_prices(12, 2)
    base = simulate(px, rf, {3: TARGET}, 10).cost.sum()
    assert simulate(px, rf, {3: TARGET}, 20).cost.sum() == pytest.approx(2 * base)
    assert simulate(px, rf, {3: TARGET}, 30).cost.sum() == pytest.approx(3 * base)
