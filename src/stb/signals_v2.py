"""v2 signals: average of return signs over several lookbacks. Reads only observations at index <= i."""
from __future__ import annotations

import numpy as np

SESSIONS_PER_MONTH = 21


def ensemble_signal(px: np.ndarray, i: int, lookbacks_months: list[int], skip_sessions: int) -> np.ndarray:
    """Mean over L of sign(P[i-skip] / P[i-21L] - 1). Values in [-1, 1]; NaN where any price is missing."""
    signs = [np.sign(px[i - skip_sessions] / px[i - SESSIONS_PER_MONTH * lb] - 1.0) for lb in lookbacks_months]
    return np.mean(signs, axis=0)


def warmup(lookbacks_months: list[int]) -> int:
    return SESSIONS_PER_MONTH * max(lookbacks_months)
