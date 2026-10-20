"""v2 data: point-in-time ETF panel on SPY's calendar and a T-bill cash rate (FRED DTB3)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW2 = ROOT / "data" / "raw_v2"
MANIFEST2 = ROOT / "data" / "manifest_v2.json"
CONFIG2 = ROOT / "config" / "v2.json"


def load_config2() -> dict:
    return json.loads(CONFIG2.read_text())


def write_raw_v2(series: pd.Series, rec: dict, name: str | None = None) -> dict:
    RAW2.mkdir(parents=True, exist_ok=True)
    path = RAW2 / f"{name or rec['ticker']}.csv"
    series.to_csv(path, header=["value"], index_label="date")
    rec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rec


def verify_manifest_v2() -> None:
    manifest = json.loads(MANIFEST2.read_text())
    for rec in manifest["series"]:
        path = RAW2 / f"{rec['file']}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != rec["sha256"]:
            raise RuntimeError(f"raw v2 data changed since download: {rec['file']}")


def _read(name: str) -> pd.Series:
    return pd.read_csv(RAW2 / f"{name}.csv", parse_dates=["date"], index_col="date")["value"]


def build_panel(tickers: list[str], max_gap: int = 2, max_total_gap_days: int = 5) -> tuple[pd.DataFrame, dict]:
    """Adjusted closes on SPY's calendar. NaN before listing. Short gaps are forward-filled and logged;
    tickers with longer or too many gaps are excluded and logged."""
    cal = _read("SPY").index
    cols, log = {}, {"excluded": {}, "gap_days": {}}
    for t in tickers:
        s = _read(t).reindex(cal)
        first, last = s.first_valid_index(), s.last_valid_index()
        live = s.loc[first:last]
        gaps = int(live.isna().sum())
        log["gap_days"][t] = gaps
        if gaps:
            filled = live.ffill(limit=max_gap)
            if filled.isna().any() or gaps > max_total_gap_days:
                log["excluded"][t] = f"{gaps} gap days (limit {max_gap} in a row, {max_total_gap_days} total)"
                continue
            s.loc[first:last] = filled
        cols[t] = s
    return pd.DataFrame(cols), log


def cash_returns(index: pd.DatetimeIndex) -> pd.Series:
    """Daily cash return from the previous available DTB3 observation (known at t-1)."""
    dtb = _read("DTB3")
    daily = dtb.reindex(dtb.index.union(index)).sort_index().ffill()
    return (daily.shift(1).reindex(index) / 100.0 / 252.0).bfill()


def tradable_mask(panel: pd.DataFrame, min_history: int, vol_window: int) -> pd.DataFrame:
    """True once an asset has `min_history` price observations (and at least vol_window + 1) up to and including t."""
    need = max(min_history, vol_window + 1)
    return panel.notna().cumsum() >= need
