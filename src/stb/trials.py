"""Append-only trial ledger. Every run is logged, including reruns and cost variants."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[2] / "trials" / "ledger.csv"
FIELDS = ["logged_utc", "stage", "study", "lookback_months", "target_vol", "cost_mult", "segment", "sharpe", "note"]


def log_trial(stage: str, lookback_months: int, target_vol: float, cost_mult: float, segment: str, sharpe: float, note: str = "", study: str = "primary") -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with LEDGER.open("a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "logged_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stage": stage, "study": study,
            "lookback_months": lookback_months, "target_vol": target_vol, "cost_mult": cost_mult,
            "segment": segment, "sharpe": f"{sharpe:.6f}", "note": note,
        })


def unique_trial_count(study: str = "primary") -> int:
    """Unique (lookback, vol target, cost multiplier) combinations registered so far."""
    if not LEDGER.exists():
        return 0
    with LEDGER.open() as fh:
        return len({(r["lookback_months"], r["target_vol"], r["cost_mult"]) for r in csv.DictReader(fh) if r["study"] == study})
