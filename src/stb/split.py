"""Chronological 60/20/20 split from the session calendar only (never touches returns)."""
from __future__ import annotations

import pandas as pd


def compute_split(index: pd.DatetimeIndex, eval_start: str, train: float, validation: float) -> dict:
    """Boundaries snap to the last session of the month that contains the cut."""
    sessions = index[index >= pd.Timestamp(eval_start)]
    n = len(sessions)

    def snap(frac: float) -> pd.Timestamp:
        cut = sessions[int(n * frac) - 1]
        in_month = sessions[(sessions.year == cut.year) & (sessions.month == cut.month)]
        return in_month[-1]

    train_end = snap(train)
    val_end = snap(train + validation)
    after_train = sessions[sessions > train_end]
    after_val = sessions[sessions > val_end]
    return {
        "eval_start": sessions[0].strftime("%Y-%m-%d"),
        "train": [sessions[0].strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")],
        "validation": [after_train[0].strftime("%Y-%m-%d"), val_end.strftime("%Y-%m-%d")],
        "test": [after_val[0].strftime("%Y-%m-%d"), sessions[-1].strftime("%Y-%m-%d")],
        "sessions": {
            "total": n,
            "train": int((sessions <= train_end).sum()),
            "validation": int(((sessions > train_end) & (sessions <= val_end)).sum()),
            "test": int((sessions > val_end).sum()),
        },
    }
