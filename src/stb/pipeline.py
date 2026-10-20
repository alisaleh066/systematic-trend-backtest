"""Shared glue for the dev, sealed-test and report scripts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stb.data import load_prices, verify_manifest
from stb.engine import Result, run_strategy
from stb.metrics import clip_trades, daily_sharpe, window_stats

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    return json.loads((ROOT / "config" / "primary.json").read_text())


def load_split() -> dict:
    return json.loads((ROOT / "config" / "split.json").read_text())


def load_inputs(cfg: dict, end: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Risk-asset prices and BIL daily returns. With `end`, later rows are dropped before anything is computed."""
    verify_manifest()
    prices = load_prices(cfg["universe"] + [cfg["cash"]])
    if end is not None:
        prices = prices.loc[:end]
    return prices[cfg["universe"]], prices[cfg["cash"]].pct_change().fillna(0.0)


def segment_stats(res: Result, rf: pd.Series, start: str, end: str) -> dict:
    return window_stats(res.net.loc[start:end], rf, clip_trades(res.trades, start, end), res.turnover.loc[start:end])


def excess_daily_sharpe(res: Result, rf: pd.Series, start: str, end: str) -> float:
    ret = res.net.loc[start:end]
    return daily_sharpe((ret - rf.loc[ret.index]).to_numpy())


def run_grid(px: pd.DataFrame, rf: pd.Series, cfg: dict, cost_bps: float) -> dict[tuple[int, float], Result]:
    out = {}
    for lb in cfg["grid"]["lookback_months"]:
        for tv in cfg["grid"]["target_vol"]:
            out[(lb, tv)] = run_strategy(px, rf, lb, tv, cost_bps, cfg["sizing"]["gross_cap"], cfg["costs"]["borrow_bps_per_year"])
    return out


def base_cost_bps(cfg: dict) -> float:
    return cfg["costs"]["commission_bps"] + cfg["costs"]["slippage_bps"]
