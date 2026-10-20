"""v2 engine: point-in-time universe on top of the v1 simulator (engine.simulate is reused unchanged)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stb.engine import Result, simulate
from stb.signals import month_end_indices
from stb.signals_v2 import ensemble_signal, warmup
from stb.universe import tradable_mask


def target_weights_v2(sig: np.ndarray, cov: np.ndarray, target_vol: float, gross_cap: float, annualisation: int = 252) -> np.ndarray:
    """w_i = s_i * (target / vol_i) / N, scaled to the ex-ante portfolio vol target, then gross-capped."""
    n = len(sig)
    vols = np.sqrt(np.diag(cov) * annualisation)
    safe = np.where(vols > 0, vols, 1.0)
    raw = np.where(vols > 0, sig * target_vol / safe / n, 0.0)
    port_vol = float(np.sqrt(raw @ cov @ raw * annualisation))
    if port_vol <= 0:
        return np.zeros(n)
    w = raw * (target_vol / port_vol)
    gross = float(np.abs(w).sum())
    return w * (gross_cap / gross) if gross > gross_cap else w


def build_fills_v2(panel: pd.DataFrame, columns: list[str], signal: dict, vol_window: int, target_vol: float, gross_cap: float,
                   start: str, min_history: int) -> dict[int, np.ndarray]:
    """{fill index: weights over `columns`}. Month-end signal at i fills at i+1; fills only from `start` on.
    An asset is traded only once tradable; weights elsewhere are 0. Uses data <= i."""
    sub = panel[columns]
    p = sub.to_numpy(float)
    ret = np.vstack([np.full(p.shape[1], np.nan), p[1:] / p[:-1] - 1.0])
    ok = tradable_mask(sub, min_history, vol_window).to_numpy()
    first_fill = int(panel.index.searchsorted(pd.Timestamp(start)))
    need = max(warmup(signal["lookbacks_months"]), vol_window)
    fills = {}
    for i in month_end_indices(panel.index):
        if i + 1 < first_fill or i < need or i + 1 >= len(panel):
            continue
        sig = ensemble_signal(p, i, signal["lookbacks_months"], signal["skip_sessions"])
        live = ok[i] & np.isfinite(sig)
        w = np.zeros(len(columns))
        if live.sum() >= 1:
            cov = np.atleast_2d(np.cov(ret[i - vol_window + 1 : i + 1][:, live], rowvar=False))
            w[live] = target_weights_v2(sig[live], cov, target_vol, gross_cap)
        fills[i + 1] = w
    return fills


def run_design(panel: pd.DataFrame, rf: pd.Series, columns: list[str], signal: dict, vol_window: int, target_vol: float,
               cost_bps: float, start: str, min_history: int, gross_cap: float = 2.0, borrow_bps: float = 25.0) -> Result:
    fills = build_fills_v2(panel, columns, signal, vol_window, target_vol, gross_cap, start, min_history)
    px = panel[columns].bfill()  # pre-listing prices are never held (weight 0); bfill only keeps returns finite
    return simulate(px, rf, fills, cost_bps, borrow_bps)


def split_dev(index: pd.DatetimeIndex, start: str, train_frac: float) -> dict:
    """75/25 chronological split of the development sessions, boundary snapped to month-end. Calendar only."""
    sessions = index[index >= pd.Timestamp(start)]
    cut = sessions[int(len(sessions) * train_frac) - 1]
    train_end = sessions[(sessions.year == cut.year) & (sessions.month == cut.month)][-1]
    after = sessions[sessions > train_end]
    return {"train": [sessions[0].strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")],
            "validation": [after[0].strftime("%Y-%m-%d"), sessions[-1].strftime("%Y-%m-%d")],
            "sessions": {"train": int((sessions <= train_end).sum()), "validation": int(len(after))}}
