"""Signals and risk estimates. Every function reads only observations at index <= i."""
from __future__ import annotations

import numpy as np

SESSIONS_PER_MONTH = 21


def momentum(px: np.ndarray, i: int, lookback_months: int) -> np.ndarray:
    """P[i-21] / P[i-21*L] - 1 for every asset (12-1 momentum when L=12)."""
    return px[i - SESSIONS_PER_MONTH] / px[i - SESSIONS_PER_MONTH * lookback_months] - 1.0


def window_cov(ret: np.ndarray, i: int, window: int) -> np.ndarray:
    """Sample covariance of daily returns over the `window` sessions ending at i (inclusive)."""
    return np.cov(ret[i - window + 1 : i + 1], rowvar=False)


def month_end_indices(index) -> list[int]:
    """Positions of the last session of each calendar month."""
    ym = index.year * 100 + index.month
    return [k for k in range(len(index)) if k == len(index) - 1 or ym[k] != ym[k + 1]]
