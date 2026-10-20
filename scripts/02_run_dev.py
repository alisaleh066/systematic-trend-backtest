"""Dev run on train + validation only. Data after the validation end is dropped before any computation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from stb import pipeline as pl  # noqa: E402
from stb.engine import run_benchmark, run_strategy  # noqa: E402
from stb.trials import log_trial  # noqa: E402

cfg, split = pl.load_config(), pl.load_split()
tr, va = split["train"], split["validation"]
dev_start, dev_end = tr[0], va[1]
px, rf = pl.load_inputs(cfg, end=dev_end)
assert px.index[-1] <= __import__("pandas").Timestamp(dev_end), "test data leaked into the dev run"

bps = pl.base_cost_bps(cfg)
grid = pl.run_grid(px, rf, cfg, bps)
cells, trial_sharpes = {}, []
for (lb, tv), res in grid.items():
    s_tr = pl.segment_stats(res, rf, *tr)
    s_va = pl.segment_stats(res, rf, *va)
    d = pl.excess_daily_sharpe(res, rf, dev_start, dev_end)
    trial_sharpes.append(d)
    cells[f"{lb}|{tv}"] = {"train_sharpe": s_tr["sharpe"], "validation_sharpe": s_va["sharpe"], "dev_daily_sharpe": d}
    log_trial("dev", lb, tv, 1, "train", s_tr["sharpe"])
    log_trial("dev", lb, tv, 1, "validation", s_va["sharpe"])

p_lb, p_tv = cfg["signal"]["lookback_months"], cfg["sizing"]["target_vol"]
primary = grid[(p_lb, p_tv)]
summary = {
    "dev_window": [dev_start, dev_end],
    "primary": {"train": pl.segment_stats(primary, rf, *tr), "validation": pl.segment_stats(primary, rf, *va)},
    "cost_stress_validation": {},
    "benchmarks_validation": {},
    "grid": cells,
    "trial_daily_sharpes_dev": trial_sharpes,
    "positive_validation_share": float(np.mean([c["validation_sharpe"] > 0 for c in cells.values()])),
}
for m in cfg["costs"]["stress_multipliers"]:
    res = run_strategy(px, rf, p_lb, p_tv, bps * m, cfg["sizing"]["gross_cap"], cfg["costs"]["borrow_bps_per_year"])
    s = pl.segment_stats(res, rf, *va)
    summary["cost_stress_validation"][f"{m}x"] = s
    log_trial("dev", p_lb, p_tv, m, "validation", s["sharpe"], "cost stress")
for name, w in cfg["benchmarks"].items():
    b = run_benchmark(px, rf, w, va[0], bps, monthly_rebalance=(name == "sixty_forty"))
    from stb.metrics import window_stats  # noqa: E402
    summary["benchmarks_validation"][name] = window_stats(b.net.loc[va[0]:va[1]], rf)

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "dev_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

for seg, key in (("train", "train_sharpe"), ("validation", "validation_sharpe")):
    lbs, tvs = cfg["grid"]["lookback_months"], cfg["grid"]["target_vol"]
    z = np.array([[cells[f"{lb}|{tv}"][key] for tv in tvs] for lb in lbs])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    im = ax.imshow(z, cmap="RdYlGn", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(tvs)), [f"{v:.0%}" for v in tvs])
    ax.set_yticks(range(len(lbs)), [f"{v}m" for v in lbs])
    ax.set_xlabel("vol target")
    ax.set_ylabel("lookback (skip last month)")
    ax.set_title(f"Sharpe, {seg}")
    for a in range(z.shape[0]):
        for b in range(z.shape[1]):
            ax.text(b, a, f"{z[a, b]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / f"heatmap_{seg}.png", dpi=130)
    plt.close(fig)

p = summary["primary"]
print(f"primary 12-1 / 10%: train Sharpe {p['train']['sharpe']:.2f}  validation Sharpe {p['validation']['sharpe']:.2f}")
print(f"grid cells with positive validation Sharpe: {summary['positive_validation_share']:.0%}")
