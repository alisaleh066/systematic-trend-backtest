"""Performance statistics and the stationary block bootstrap."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

ANN = 252


def max_drawdown(ret: pd.Series) -> float:
    equity = (1 + ret).cumprod()
    equity = pd.concat([pd.Series([1.0]), equity.reset_index(drop=True)])
    return float((equity / equity.cummax() - 1).min())


def daily_sharpe(excess: np.ndarray) -> float:
    sd = np.std(excess, ddof=1)
    return float(np.mean(excess) / sd) if sd > 1e-12 else float("nan")


def sharpe(ret: pd.Series, rf: pd.Series) -> float:
    return daily_sharpe((ret - rf.reindex(ret.index)).to_numpy()) * math.sqrt(ANN)


def window_stats(ret: pd.Series, rf: pd.Series, trades: pd.DataFrame | None = None, turnover: pd.Series | None = None) -> dict:
    """Headline statistics for one window. `trades` must already be clipped to the window."""
    ex = (ret - rf.reindex(ret.index)).to_numpy()
    n = len(ret)
    ann_ret = float((1 + ret).prod() ** (ANN / n) - 1)
    down = math.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    mdd = max_drawdown(ret)
    out = {
        "start": ret.index[0].strftime("%Y-%m-%d"),
        "end": ret.index[-1].strftime("%Y-%m-%d"),
        "sessions": n,
        "total_return": float((1 + ret).prod() - 1),
        "ann_return": ann_ret,
        "ann_vol": float(ret.std(ddof=1) * math.sqrt(ANN)),
        "sharpe": daily_sharpe(ex) * math.sqrt(ANN),
        "sortino": float(np.mean(ex) * ANN / (down * math.sqrt(ANN))) if down > 1e-12 else None,
        "max_drawdown": mdd,
        "calmar": ann_ret / abs(mdd) if mdd < 0 else None,
    }
    if turnover is not None:
        out["turnover_per_year"] = float(turnover.sum() / (n / ANN))
    if trades is not None:
        pnl = trades["net_ret"]
        wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        out.update(
            round_trips=int(len(trades)),
            hit_rate=float((pnl > 0).mean()) if len(trades) else None,
            profit_factor=float(wins / losses) if losses > 0 else None,
        )
    return out


def clip_trades(trades: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Completed trades whose exit is inside [start, end]; a trade opened earlier counts as entered at `start`."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    t = trades[(trades["exit"] >= s) & (trades["exit"] <= e)].copy()
    t["entry"] = t["entry"].clip(lower=s)
    return t


def stationary_bootstrap_sharpe_ci(excess: np.ndarray, mean_block: int, resamples: int, seed: int) -> tuple[float, float]:
    """95% CI for the annualised Sharpe (Politis-Romano stationary bootstrap)."""
    rng = np.random.default_rng(seed)
    n = len(excess)
    p = 1.0 / mean_block
    idx = np.empty((resamples, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, resamples)
    restart = rng.random((resamples, n)) < p
    fresh = rng.integers(0, n, (resamples, n))
    for k in range(1, n):
        idx[:, k] = np.where(restart[:, k], fresh[:, k], (idx[:, k - 1] + 1) % n)
    sample = excess[idx]
    sd = sample.std(axis=1, ddof=1)
    sr = np.where(sd > 1e-12, sample.mean(axis=1) / np.where(sd > 1e-12, sd, 1.0), np.nan) * math.sqrt(ANN)
    lo, hi = np.nanpercentile(sr, [2.5, 97.5])
    return float(lo), float(hi)
