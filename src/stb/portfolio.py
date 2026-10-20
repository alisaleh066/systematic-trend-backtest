"""Volatility-targeted long/short weights."""
from __future__ import annotations

import numpy as np


def target_weights(mom: np.ndarray, cov: np.ndarray, target_vol: float, gross_cap: float, annualisation: int = 252) -> np.ndarray:
    """w_i = sign_i * (target / vol_i) / N, scaled to the ex-ante portfolio vol target, then gross-capped."""
    n = len(mom)
    vols = np.sqrt(np.diag(cov) * annualisation)
    sign = np.sign(mom)
    raw = np.where(vols > 0, sign * target_vol / np.where(vols > 0, vols, 1.0) / n, 0.0)
    port_vol = float(np.sqrt(raw @ cov @ raw * annualisation))
    if port_vol <= 0:
        return np.zeros(n)
    w = raw * (target_vol / port_vol)
    gross = float(np.abs(w).sum())
    if gross > gross_cap:
        w *= gross_cap / gross
    return w
