"""Download, log and load free daily price data (yfinance, Stooq fallback)."""
from __future__ import annotations

import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.json"


def last_complete_month_end_exclusive(today: datetime | None = None) -> str:
    """First day of the current month: yfinance `end` is exclusive."""
    today = today or datetime.now(timezone.utc)
    return today.strftime("%Y-%m-01")


def _from_yfinance(ticker: str, start: str, end: str) -> pd.Series:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, multi_level_index=False)
    if df is None or df.empty:
        raise ValueError(f"yfinance returned no rows for {ticker}")
    return df["Close"].dropna().rename(ticker)


def _from_stooq(ticker: str, start: str, end: str) -> pd.Series:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    text = urlopen(url, timeout=30).read().decode()
    df = pd.read_csv(io.StringIO(text), parse_dates=["Date"], index_col="Date")
    if df.empty or "Close" not in df:
        raise ValueError(f"stooq returned no rows for {ticker}")
    df = df.loc[start : pd.Timestamp(end) - pd.Timedelta(days=1)]
    return df["Close"].dropna().rename(ticker)


def download_ticker(ticker: str, start: str, end: str, retries: int = 3) -> tuple[pd.Series, dict]:
    """Return (adjusted close, manifest record). Tries yfinance, then Stooq."""
    errors: list[str] = []
    for source, fn, url in (
        ("yfinance", _from_yfinance, f"yfinance:{ticker}"),
        ("stooq", _from_stooq, f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"),
    ):
        for attempt in range(retries):
            try:
                s = fn(ticker, start, end)
                s.index = pd.to_datetime(s.index).tz_localize(None)
                note = "dividend+split adjusted" if source == "yfinance" else "split-adjusted only (dividends NOT included)"
                rec = {
                    "ticker": ticker,
                    "source": source,
                    "url": url,
                    "adjustment": note,
                    "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "rows": int(len(s)),
                    "first": s.index[0].strftime("%Y-%m-%d"),
                    "last": s.index[-1].strftime("%Y-%m-%d"),
                    "fallback_errors": errors,
                }
                return s, rec
            except Exception as exc:  # noqa: BLE001 - logged, then fall through to next source
                errors.append(f"{source}#{attempt + 1}: {type(exc).__name__}: {exc}")
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"all sources failed for {ticker}: {errors}")


def write_raw(series: pd.Series, rec: dict) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{rec['ticker']}.csv"
    series.to_csv(path, header=["adj_close"], index_label="date")
    rec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rec


def write_manifest(records: list[dict], start: str, end_exclusive: str) -> None:
    MANIFEST.write_text(
        json.dumps({"start": start, "end_exclusive": end_exclusive, "tickers": records}, indent=2) + "\n"
    )


def load_prices(tickers: list[str]) -> pd.DataFrame:
    """Adjusted closes on the intersection of sessions. No forward-fill."""
    cols = [pd.read_csv(RAW / f"{t}.csv", parse_dates=["date"], index_col="date")["adj_close"].rename(t) for t in tickers]
    return pd.concat(cols, axis=1, join="inner").sort_index()


def verify_manifest() -> None:
    """Raise if any raw file no longer matches the logged sha256."""
    manifest = json.loads(MANIFEST.read_text())
    for rec in manifest["tickers"]:
        digest = hashlib.sha256((RAW / f"{rec['ticker']}.csv").read_bytes()).hexdigest()
        if digest != rec["sha256"]:
            raise RuntimeError(f"raw data changed since download: {rec['ticker']}")
