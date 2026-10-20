"""Daily portfolio simulator: signal at close t, fill at close t+1, costs and borrow charged explicitly."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stb.portfolio import target_weights
from stb.signals import SESSIONS_PER_MONTH, month_end_indices, momentum, window_cov

EPS = 1e-9


@dataclass
class Result:
    net: pd.Series  # daily net return
    cost: pd.Series  # daily transaction cost (fraction of equity)
    borrow: pd.Series
    turnover: pd.Series  # sum |trade| on fill days, else 0
    gross: pd.Series  # gross exposure held at the close
    weights: pd.DataFrame  # weights held at the close
    trades: pd.DataFrame  # completed round trips


def build_fills(px: pd.DataFrame, lookback_months: int, target_vol: float, gross_cap: float = 2.0, vol_window: int = 63) -> dict[int, np.ndarray]:
    """{fill index: target weights}. Month-end signal at i is filled at i+1; uses data <= i only."""
    p = px.to_numpy(float)
    ret = np.vstack([np.full(p.shape[1], np.nan), p[1:] / p[:-1] - 1.0])
    first = max(SESSIONS_PER_MONTH * lookback_months, vol_window)
    fills = {}
    for i in month_end_indices(px.index):
        if i < first or i + 1 >= len(px):
            continue
        fills[i + 1] = target_weights(momentum(p, i, lookback_months), window_cov(ret, i, vol_window), target_vol, gross_cap)
    return fills


def simulate(px: pd.DataFrame, rf: pd.Series, fills: dict[int, np.ndarray], cost_bps: float, borrow_bps: float = 25.0) -> Result:
    p = px.to_numpy(float)
    t, n = p.shape
    ret = np.vstack([np.zeros(n), p[1:] / p[:-1] - 1.0])
    f = rf.reindex(px.index).to_numpy(float)
    rate, borrow_daily = cost_bps / 1e4, borrow_bps / 1e4 / 252

    net, cost_s, borrow_s, turn = (np.zeros(t) for _ in range(4))
    wts = np.zeros((t, n))
    w = np.zeros(n)
    open_pos: dict[int, tuple[int, int]] = {}  # asset -> (sign, entry index)
    trades = []

    for d in range(1, t):
        gross_pf = f[d] + w @ (ret[d] - f[d])
        borrow = borrow_daily * float(np.maximum(-w, 0).sum())
        w_drift = w * (1 + ret[d]) / (1 + gross_pf)
        cost = 0.0
        if d in fills:
            target = fills[d]
            turn[d] = float(np.abs(target - w_drift).sum())
            cost = rate * turn[d]
            for j in range(n):
                old = int(np.sign(w_drift[j])) if abs(w_drift[j]) > EPS else 0
                new = int(np.sign(target[j])) if abs(target[j]) > EPS else 0
                if old == new:
                    continue
                if old != 0:
                    s, e = open_pos.pop(j)
                    gross = s * (p[d, j] / p[e, j] - 1.0)
                    trades.append((px.columns[j], s, px.index[e], px.index[d], gross, gross - 2 * rate))
                if new != 0:
                    open_pos[j] = (new, d)
            w_drift = target
        net[d] = gross_pf - borrow - cost
        cost_s[d], borrow_s[d] = cost, borrow
        w = w_drift
        wts[d] = w

    idx = px.index
    return Result(
        net=pd.Series(net, idx),
        cost=pd.Series(cost_s, idx),
        borrow=pd.Series(borrow_s, idx),
        turnover=pd.Series(turn, idx),
        gross=pd.Series(np.abs(wts).sum(axis=1), idx),
        weights=pd.DataFrame(wts, idx, px.columns),
        trades=pd.DataFrame(trades, columns=["asset", "sign", "entry", "exit", "gross_ret", "net_ret"]),
    )


def run_strategy(px: pd.DataFrame, rf: pd.Series, lookback_months: int, target_vol: float, cost_bps: float, gross_cap: float = 2.0, borrow_bps: float = 25.0) -> Result:
    return simulate(px, rf, build_fills(px, lookback_months, target_vol, gross_cap), cost_bps, borrow_bps)


def run_benchmark(px: pd.DataFrame, rf: pd.Series, weights: dict[str, float], start: str, cost_bps: float, monthly_rebalance: bool) -> Result:
    """Buy at the close before `start` (exposure begins on `start`), same costs. The entry cost is
    booked on the first session of the window so it is not lost outside it."""
    w = np.array([weights.get(c, 0.0) for c in px.columns])
    s0 = int(px.index.searchsorted(pd.Timestamp(start))) - 1
    fills = {s0: w}
    if monthly_rebalance:
        fills.update({i + 1: w for i in month_end_indices(px.index) if s0 < i + 1 < len(px)})
    res = simulate(px, rf, fills, cost_bps)
    res.net.iloc[s0 + 1] -= res.cost.iloc[s0]
    res.cost.iloc[s0 + 1] += res.cost.iloc[s0]
    res.cost.iloc[s0] = 0.0
    res.net.iloc[s0] = 0.0
    return res
