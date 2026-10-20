"""SPY regimes per contract lines 70-76. Each day's label uses data up to the previous close only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def efficiency_ratio(close: pd.Series, window: int = 63) -> pd.Series:
    """ER(t) = |C[t-1] - C[t-1-window]| / sum_{j=1..window} |C[t-j] - C[t-j-1]|. NaN on a zero denominator."""
    step = close.diff().abs()
    denom = step.rolling(window).sum().shift(1)
    num = (close.shift(1) - close.shift(1 + window)).abs()
    er = num / denom.where(denom > 0)
    return er


def vol_regime(close: pd.Series, vol_window: int = 20, lookback: int = 252) -> pd.Series:
    """True = high vol: preceding 20-session realised vol above the median of the 252 earlier estimates."""
    vol = close.pct_change().rolling(vol_window).std().shift(1)
    median = vol.shift(1).rolling(lookback).median()
    out = pd.Series(np.where(vol > median, "high", "low"), close.index)
    return out.where(vol.notna() & median.notna(), "unknown")


def regime_labels(close: pd.Series, er_threshold: float = 0.30, er_window: int = 63, vol_window: int = 20, lookback: int = 252) -> pd.DataFrame:
    er = efficiency_ratio(close, er_window)
    trend = pd.Series(np.where(er >= er_threshold, "trend", "range"), close.index).where(er.notna(), "unknown")
    vol = vol_regime(close, vol_window, lookback)
    cell = (trend + "/" + vol).where((trend != "unknown") & (vol != "unknown"), "unknown")
    return pd.DataFrame({"trend": trend, "vol": vol, "cell": cell})


def cell_coverage(labels: pd.DataFrame, trades: pd.DataFrame, min_trades: int) -> dict:
    """Round trips per joint cell by entry date; cells under `min_trades` are flagged sparse."""
    cells = ["trend/high", "trend/low", "range/high", "range/low"]
    entry_cell = labels["cell"].reindex(trades["entry"]).to_numpy() if len(trades) else np.array([])
    counts = {c: int((entry_cell == c).sum()) for c in cells}
    return {"trades": counts, "sparse": [c for c, n in counts.items() if n < min_trades], "unknown": int((entry_cell == "unknown").sum())}


def decompose(ret: pd.Series, labels: pd.DataFrame) -> dict:
    """Mean annualised return and day count by calendar year, trend/range, vol level and joint cell."""
    def agg(g):
        return {"days": int(len(g)), "ann_return": float(g.mean() * 252)}
    lab = labels.reindex(ret.index)
    out = {"year": {str(y): agg(g) for y, g in ret.groupby(ret.index.year)}}
    for key in ("trend", "vol", "cell"):
        out[key] = {str(k): agg(g) for k, g in ret.groupby(lab[key])}
    return out
