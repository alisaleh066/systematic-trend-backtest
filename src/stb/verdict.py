"""Overfitting flags and the PROMOTE / WATCH / REJECT verdict (contract lines 109-113, 152)."""
from __future__ import annotations

import numpy as np


def neighbour_gap(cells: dict[tuple[int, float], float], primary: tuple[int, float], lookbacks: list, vols: list) -> float:
    """Primary Sharpe minus the median Sharpe of its adjacent grid cells (including diagonals)."""
    a, b = lookbacks.index(primary[0]), vols.index(primary[1])
    nb = [cells[(lookbacks[i], vols[j])] for i in range(a - 1, a + 2) for j in range(b - 1, b + 2)
          if 0 <= i < len(lookbacks) and 0 <= j < len(vols) and (i, j) != (a, b)]
    return float(cells[primary] - np.median(nb))


def overfit_flags(f: dict, oos_sharpe: float, dsr: dict | None, positive_validation_share: float, gap_validation: float, gap_oos: float, trips: int) -> list[str]:
    flags = []
    if oos_sharpe > f["max_sharpe"]:
        flags.append(f"OOS Sharpe {oos_sharpe:.2f} > {f['max_sharpe']}")
    if dsr is None:
        flags.append("Deflated Sharpe unavailable")
    elif dsr["probability"] < f["min_dsr_prob"]:
        flags.append(f"Deflated Sharpe probability {dsr['probability']:.2f} < {f['min_dsr_prob']}")
    if positive_validation_share < f["min_positive_grid_share"]:
        flags.append(f"only {positive_validation_share:.0%} of grid cells have positive validation Sharpe")
    for name, gap in (("validation", gap_validation), ("OOS", gap_oos)):
        if gap > f["neighbour_median_gap"]:
            flags.append(f"result hinges on one setting ({name}: primary beats neighbour median by {gap:.2f})")
    if trips < f["min_oos_trades"]:
        flags.append(f"only {trips} OOS round trips < {f['min_oos_trades']}")
    return flags


def decide(oos_1x: dict, oos_2x: dict, ci_low: float, complete_oos_years: int, sparse_cells: list[str], min_trips: int) -> dict:
    gates = {
        "net OOS return > 0 at 1x costs": oos_1x["total_return"] > 0,
        "net OOS return > 0 at 2x costs": oos_2x["total_return"] > 0,
        "profit factor > 1 at 1x costs": (oos_1x.get("profit_factor") or 0) > 1,
        "profit factor > 1 at 2x costs": (oos_2x.get("profit_factor") or 0) > 1,
        "bootstrap 95% CI for Sharpe above 0": ci_low > 0,
        f">= {min_trips} OOS round trips": oos_1x["round_trips"] >= min_trips,
        ">= 2 complete OOS calendar years": complete_oos_years >= 2,
        "every joint regime cell has enough trades": not sparse_cells,
    }
    if oos_1x["total_return"] <= 0:
        verdict = "REJECT"  # negative net OOS economics under the agreed cost model
    elif all(gates.values()):
        verdict = "PROMOTE"
    else:
        verdict = "WATCH"
    return {"verdict": verdict, "gates": gates, "failed": [k for k, v in gates.items() if not v]}
