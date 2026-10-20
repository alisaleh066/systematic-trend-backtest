"""Intermarket radar break rules, ported unchanged from Financedashboard/server/routine_radar.py:46-83,
split into the three registered rule types, plus a month-clustered stationary bootstrap for break-vs-hold tests."""
from __future__ import annotations

import statistics

import numpy as np


def _finite(v) -> bool:
    return v is not None and np.isfinite(v)


def _correlation(x, y):
    try:
        return statistics.correlation(x, y)
    except statistics.StatisticsError:
        return None


def _changes(values, difference=False):
    return [b - a if difference else b / a - 1 for a, b in zip(values, values[1:])]


def classify_rules(x, y, expected=1, left_difference=False, right_difference=False, ratio_confirmation=False) -> dict | None:
    """Per-rule break flags on the last 313 aligned observations, or None for 'no read'."""
    x, y = list(x), list(y)
    if len(x) != len(y) or len(x) < 313 or any(not _finite(v) for v in x[-313:] + y[-313:]):
        return None
    x, y = x[-313:], y[-313:]
    if (not left_difference and min(x) <= 0) or (not right_difference and min(y) <= 0):
        return None
    dx, dy = _changes(x, left_difference), _changes(y, right_difference)
    rolling = [_correlation(dx[i - 60 : i], dy[i - 60 : i]) for i in range(60, len(dx) + 1)]
    if any(v is None for v in rolling):
        return None
    corr, previous, baseline = rolling[-1], rolling[-2], statistics.mean(rolling[-253:-1])
    correlation_break = corr * previous < 0 or abs(corr - baseline) > 0.5

    yy = [expected * v for v in y]

    def extreme(a):
        return 1 if a[-1] > max(a[-21:-1]) else -1 if a[-1] < min(a[-21:-1]) else 0

    a, b = extreme(x), extreme(yy)
    divergence = bool((a and a != b) or (b and b != a))
    ratios = [p / q for p, q in zip(x[-252:], y[-252:])] if min(y[-252:]) > 0 or max(y[-252:]) < 0 else []
    if ratio_confirmation and ratios and extreme(y) and extreme(ratios) != extreme(y):
        divergence = True
    spread = statistics.pstdev(ratios) if ratios else 0
    z = (ratios[-1] - statistics.mean(ratios)) / spread if spread > 0 else None
    z_break = z is not None and abs(z) > 2
    any_break = divergence or correlation_break or z_break
    if not any_break and z is None:
        return None  # original returns 'no read': holds cannot be established without a ratio z-score
    return {"divergence": divergence, "correlation": bool(correlation_break), "zscore": bool(z_break), "any_break": bool(any_break)}


def stationary_month_indices(n_months: int, mean_block: int, resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.empty((resamples, n_months), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_months, resamples)
    restart = rng.random((resamples, n_months)) < 1.0 / mean_block
    fresh = rng.integers(0, n_months, (resamples, n_months))
    for k in range(1, n_months):
        idx[:, k] = np.where(restart[:, k], fresh[:, k], (idx[:, k - 1] + 1) % n_months)
    return idx


def break_vs_hold(month_id: np.ndarray, is_break: np.ndarray, outcome: np.ndarray, mean_block: int = 3, resamples: int = 10000, seed: int = 20261020) -> dict | None:
    """Mean outcome after breaks minus after holds, pooled across pairs, with a month-clustered stationary bootstrap."""
    ok = np.isfinite(outcome)
    month_id, is_break, outcome = month_id[ok], is_break[ok], outcome[ok]
    if is_break.sum() < 5 or (~is_break).sum() < 5:
        return None
    months = np.unique(month_id)
    pos = np.searchsorted(months, month_id)
    m = len(months)
    sb, nb = np.bincount(pos[is_break], outcome[is_break], m), np.bincount(pos[is_break], minlength=m).astype(float)
    sh, nh = np.bincount(pos[~is_break], outcome[~is_break], m), np.bincount(pos[~is_break], minlength=m).astype(float)
    diff = sb.sum() / nb.sum() - sh.sum() / nh.sum()
    idx = stationary_month_indices(m, mean_block, resamples, seed)
    counts = np.apply_along_axis(np.bincount, 1, idx, None, m).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = (counts @ sb) / (counts @ nb) - (counts @ sh) / (counts @ nh)
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"mean_break": float(sb.sum() / nb.sum()), "mean_hold": float(sh.sum() / nh.sum()), "diff": float(diff), "ci95": [float(lo), float(hi)],
            "p_value": float(max(p, 1.0 / (len(boot) + 1))), "n_break": int(nb.sum()), "n_hold": int(nh.sum()), "months": int(m)}


def benjamini_hochberg(pvalues: list[float], q: float = 0.05) -> list[bool]:
    order = np.argsort(pvalues)
    n = len(pvalues)
    passed = np.zeros(n, dtype=bool)
    thresh = [(k + 1) / n * q for k in range(n)]
    last = -1
    for rank, i in enumerate(order):
        if pvalues[i] <= thresh[rank]:
            last = rank
    for rank, i in enumerate(order):
        passed[i] = rank <= last
    return passed.tolist()
