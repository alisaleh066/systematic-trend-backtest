"""Deflated Sharpe ratio (Bailey and Lopez de Prado), ported from
Financedashboard/server/analysis_engine.py:94 so both repos use the same method. Sharpe inputs are per-period (daily)."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

GAMMA = 0.5772156649015329


def deflated_sharpe(excess, trial_sharpes, trial_count: int) -> dict | None:
    """Probability that the observed Sharpe beats the expected best of `trial_count` trials.
    `excess`: daily excess returns of the chosen strategy. `trial_sharpes`: daily Sharpes of the registered trials."""
    values = np.asarray(excess, float)
    if len(values) < 30 or len(trial_sharpes) < 2 or trial_count < 2 or np.std(values, ddof=1) < 1e-12:
        return None
    trial_var = float(np.var(trial_sharpes, ddof=1))
    if trial_var <= 1e-12:
        return None
    n = len(values)
    avg, sd = values.mean(), values.std(ddof=1)
    m2 = np.mean((values - avg) ** 2)
    skew = np.mean((values - avg) ** 3) / m2**1.5
    kurt = np.mean((values - avg) ** 4) / m2**2
    sr = avg / sd
    nd = NormalDist()
    expected = math.sqrt(trial_var) * ((1 - GAMMA) * nd.inv_cdf(1 - 1 / trial_count) + GAMMA * nd.inv_cdf(1 - 1 / (trial_count * math.e)))
    denom = 1 - skew * sr + ((kurt - 1) / 4) * sr * sr
    if denom <= 0:
        return None
    score = nd.cdf((sr - expected) * math.sqrt(n - 1) / math.sqrt(denom))
    return {
        "probability": float(score),
        "daily_sharpe": float(sr),
        "expected_max_daily_sharpe": float(expected),
        "trials": int(trial_count),
        "trial_sharpe_variance": trial_var,
        "skew": float(skew),
        "kurtosis": float(kurt),
    }
