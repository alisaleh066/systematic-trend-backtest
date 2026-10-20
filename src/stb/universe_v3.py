"""v3 data: long-history proxy universe (Vanguard funds, FRED/Stooq spot FX and commodities) on the VFINX calendar."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW3 = ROOT / "data" / "raw_v3"
MANIFEST3 = ROOT / "data" / "manifest_v3.json"
CONFIG3 = ROOT / "config" / "v3.json"


def load_config3() -> dict:
    return json.loads(CONFIG3.read_text())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fetch_fred(series_id: str, start: str, end_exclusive: str) -> tuple[pd.Series, dict]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    rows = list(csv.DictReader(io.StringIO(urlopen(url, timeout=60).read().decode("utf-8-sig"))))
    key = "observation_date" if "observation_date" in rows[0] else "DATE"
    vals = {pd.Timestamp(r[key]): float(r[series_id]) for r in rows if r[series_id] not in ("", ".") and r[key] < end_exclusive}
    if not vals:
        raise ValueError(f"FRED returned no rows for {series_id}")
    return pd.Series(vals).sort_index(), {"source": "FRED", "url": url}


def fetch_stooq(symbol: str, start: str, end_exclusive: str) -> tuple[pd.Series, dict]:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    text = urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read().decode()
    df = pd.read_csv(io.StringIO(text))
    if "Close" not in df or df.empty:
        raise ValueError(f"stooq returned no usable rows for {symbol}: {text[:80]!r}")
    s = pd.Series(df["Close"].to_numpy(float), pd.to_datetime(df["Date"]))
    return s.loc[start : pd.Timestamp(end_exclusive) - pd.Timedelta(days=1)].dropna(), {"source": "stooq", "url": url}


def write_raw3(name: str, series: pd.Series, meta: dict) -> dict:
    RAW3.mkdir(parents=True, exist_ok=True)
    path = RAW3 / f"{name}.csv"
    series.to_csv(path, header=["value"], index_label="date")
    return {"name": name, **meta, "downloaded_utc": _now(), "rows": int(len(series)),
            "first": series.index[0].strftime("%Y-%m-%d"), "last": series.index[-1].strftime("%Y-%m-%d"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify_manifest_v3() -> None:
    for rec in json.loads(MANIFEST3.read_text())["series"]:
        if hashlib.sha256((RAW3 / f"{rec['name']}.csv").read_bytes()).hexdigest() != rec["sha256"]:
            raise RuntimeError(f"raw v3 data changed since download: {rec['name']}")


def read_raw3(name: str) -> pd.Series:
    return pd.read_csv(RAW3 / f"{name}.csv", parse_dates=["date"], index_col="date")["value"]


def fx_usd_per_unit(raw: pd.Series, quote: str) -> pd.Series:
    """FRED quotes some pairs as foreign units per USD; invert those so the series is USD per foreign unit."""
    return 1.0 / raw if quote == "invert" else raw


def fit_to_calendar(raw: pd.Series, cal: pd.DatetimeIndex, max_gap: int) -> tuple[pd.Series | None, str | None]:
    """Reindex to the calendar. Gaps of <= max_gap sessions in a row are forward-filled; a longer gap excludes the series."""
    s = raw.reindex(cal)
    first, last = s.first_valid_index(), s.last_valid_index()
    if first is None:
        return None, "no observations on the calendar"
    if last < cal[-1] - pd.Timedelta(days=10):
        return None, f"stale: last observation {last.date()}"
    live = s.loc[first:]
    run = live.isna().astype(int).groupby(live.notna().cumsum()).sum().max()
    if run > max_gap:
        return None, f"gap of {int(run)} sessions in a row (limit {max_gap})"
    s.loc[first:] = live.ffill(limit=max_gap)
    return s, None


def excess_return_index(spot: pd.Series, rf: pd.Series) -> pd.Series:
    """Synthetic index whose return is spot move + rf, so the engine's r - rf equals the spot move. NaN before listing."""
    first = spot.first_valid_index()
    r = spot.loc[first:].pct_change()
    growth = 1.0 + r + rf.reindex(r.index).fillna(0.0)
    growth.iloc[0] = 1.0  # listing day: base level, no return yet
    return (100.0 * np.cumprod(growth)).reindex(spot.index)


def cash_returns3(cal: pd.DatetimeIndex) -> pd.Series:
    dtb = read_raw3("DTB3")
    daily = dtb.reindex(dtb.index.union(cal)).sort_index().ffill()
    return (daily.shift(1).reindex(cal) / 100.0 / 252.0).bfill()


def build_panel3(cfg: dict) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Price panel (NaN before listing), daily cash return, and a log of exclusions."""
    u = cfg["universe"]
    cal = read_raw3(u["calendar"]).index
    rf = cash_returns3(cal)
    manifest = json.loads(MANIFEST3.read_text())
    available = {r["name"] for r in manifest["series"]}
    cols, log = {}, {"excluded": dict(manifest.get("dropped", {}))}
    names = u["funds"] + list(u["stooq_spot"]) + u["fred_spot"] + list(u["fred_fx"])
    for name in names:
        if name not in available:
            continue
        raw = read_raw3(name)
        if name in u["fred_fx"]:
            raw = fx_usd_per_unit(raw, u["fred_fx"][name])
        s, why = fit_to_calendar(raw, cal, u["max_gap_ffill"])
        if s is None:
            log["excluded"][name] = why
            continue
        cols[name] = excess_return_index(s, rf) if name in u["excess_return_proxies"] else s
    return pd.DataFrame(cols, index=cal), rf, log
