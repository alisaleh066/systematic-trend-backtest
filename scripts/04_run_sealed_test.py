"""Open the sealed test set. Refuses unless freeze.json matches and the tree is clean. First run writes the
seal record; any later run must reproduce identical numbers."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stb import pipeline as pl  # noqa: E402
from stb import seal  # noqa: E402
from stb.dsr import deflated_sharpe  # noqa: E402
from stb.engine import run_benchmark, run_strategy  # noqa: E402
from stb.metrics import clip_trades, stationary_bootstrap_sharpe_ci, window_stats  # noqa: E402
from stb.regimes import cell_coverage, decompose, regime_labels  # noqa: E402
from stb.trials import log_trial  # noqa: E402
from stb.verdict import decide, neighbour_gap, overfit_flags  # noqa: E402

RES = ROOT / "results"
OPENED = RES / "seal_opened.json"

freeze = seal.check_freeze()
reproducing = OPENED.exists()
if not reproducing:
    OPENED.write_text(json.dumps({"opened_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **{k: freeze[k] for k in ("git_head", "config_sha256", "code_sha256")}}, indent=2) + "\n")

cfg, split = pl.load_config(), pl.load_split()
dev = json.loads((RES / "dev_summary.json").read_text())
te, tr, va = split["test"], split["train"], split["validation"]
px, rf = pl.load_inputs(cfg)
bps, gcap, borrow = pl.base_cost_bps(cfg), cfg["sizing"]["gross_cap"], cfg["costs"]["borrow_bps_per_year"]
p_lb, p_tv = cfg["signal"]["lookback_months"], cfg["sizing"]["target_vol"]
lbs, vols = cfg["grid"]["lookback_months"], cfg["grid"]["target_vol"]

# primary at 1x / 2x / 3x costs
runs = {m: run_strategy(px, rf, p_lb, p_tv, bps * m, gcap, borrow) for m in [1, *cfg["costs"]["stress_multipliers"]]}
oos = {f"{m}x": pl.segment_stats(r, rf, *te) for m, r in runs.items()}
primary = runs[1]
ret = primary.net.loc[te[0]:te[1]]
excess = (ret - rf.loc[ret.index]).to_numpy()

# bootstrap and deflated Sharpe
ci = stationary_bootstrap_sharpe_ci(excess, cfg["bootstrap"]["mean_block"], cfg["bootstrap"]["resamples"], cfg["bootstrap"]["seed"])
dsr = deflated_sharpe(excess, dev["trial_daily_sharpes_dev"], freeze["trial_count"])

# grid on the test set: sensitivity only
grid = pl.run_grid(px, rf, cfg, bps)
oos_cells = {k: pl.segment_stats(r, rf, *te)["sharpe"] for k, r in grid.items()}

# benchmarks, same dates, same costs
bench, bench_ret = {}, {}
for name, w in cfg["benchmarks"].items():
    b = run_benchmark(px, rf, w, te[0], bps, monthly_rebalance=(name == "sixty_forty"))
    bench_ret[name] = b.net.loc[te[0]:te[1]]
    bench[name] = window_stats(bench_ret[name], rf, turnover=b.turnover.loc[te[0]:te[1]])
    bench[name]["costs_paid"] = float(b.cost.loc[te[0]:te[1]].sum())

# crisis windows (all fall before the sealed test; labelled by segment)
def segment_of(date: str) -> str:
    d = pd.Timestamp(date)
    return "train" if d <= pd.Timestamp(tr[1]) else "validation" if d <= pd.Timestamp(va[1]) else "test"

crises = {}
for name, (a, b) in cfg["crises"].items():
    a = max(pd.Timestamp(a), pd.Timestamp(split["eval_start"])).strftime("%Y-%m-%d")
    row = {"segment": segment_of(a), "window": [a, b], "strategy": window_stats(primary.net.loc[a:b], rf)}
    for bn, br in (("spy", "spy"), ("sixty_forty", "sixty_forty")):
        full = run_benchmark(px, rf, cfg["benchmarks"][bn], a, bps, monthly_rebalance=(bn == "sixty_forty"))
        row[bn] = window_stats(full.net.loc[a:b], rf)
    crises[name] = row

# regimes on SPY, decomposition and coverage gate
labels = regime_labels(px["SPY"], cfg["regimes"]["er_threshold"], cfg["regimes"]["er_window"], cfg["regimes"]["vol_window"], cfg["regimes"]["vol_median_lookback"])
oos_trades = clip_trades(primary.trades, *te)
coverage = cell_coverage(labels, oos_trades, cfg["regimes"]["min_trades_per_cell"])
decomp = decompose(ret, labels)

# flags and verdict
t0, t1 = pd.Timestamp(te[0]), pd.Timestamp(te[1])
full_years = [y for y in range(t0.year, t1.year + 1) if pd.Timestamp(y, 1, 1) >= t0 and pd.Timestamp(y, 12, 31) <= t1]
pos_share = dev["positive_validation_share"]
val_cells = {tuple(map(float, k.split("|"))): v["validation_sharpe"] for k, v in dev["grid"].items()}
val_cells = {(int(a), b): v for (a, b), v in val_cells.items()}
gap_val = neighbour_gap(val_cells, (p_lb, p_tv), lbs, vols)
gap_oos = neighbour_gap(oos_cells, (p_lb, p_tv), lbs, vols)
flags = overfit_flags(cfg["flags"], oos["1x"]["sharpe"], dsr, pos_share, gap_val, gap_oos, oos["1x"]["round_trips"])
decision = decide(oos["1x"], oos["2x"], ci[0], len(full_years), coverage["sparse"], cfg["flags"]["min_oos_trades"])

results = {
    "strategy": "cross-asset time-series momentum, 12-1, 10% vol target, long/short, gross cap 2x",
    "oos_window": te,
    "headline_oos_1x_costs": oos["1x"],
    "oos_cost_stress": oos,
    "sharpe_bootstrap_95ci": list(ci),
    "deflated_sharpe": dsr,
    "benchmarks_oos": bench,
    "crisis_periods": crises,
    "grid_oos_sharpe": {f"{a}|{b}": v for (a, b), v in oos_cells.items()},
    "grid_validation_sharpe": {f"{a}|{b}": v for (a, b), v in val_cells.items()},
    "neighbour_gap": {"validation": gap_val, "oos": gap_oos},
    "regime_coverage": coverage,
    "regime_decomposition_oos": decomp,
    "complete_oos_years": full_years,
    "trial_count": freeze["trial_count"],
    "flags": flags,
    "verdict": decision,
    "validation_primary": freeze["validation_primary"]["validation"],
}

if reproducing:
    prior = json.loads((RES / "results.json").read_text())
    if json.dumps(prior, sort_keys=True) != json.dumps(json.loads(json.dumps(results)), sort_keys=True):
        sys.exit("REPRODUCTION FAILED: numbers differ from the first sealed run")
    print("reproduced identical results")
else:
    (RES / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    daily = pd.DataFrame({"strategy": ret, **{k: v for k, v in bench_ret.items()}, "rf": rf.loc[ret.index]})
    daily.to_csv(RES / "oos_daily.csv", index_label="date")
    oos_trades.to_csv(RES / "oos_trades.csv", index=False)
    for m in cfg["costs"]["stress_multipliers"]:
        log_trial("sealed", p_lb, p_tv, m, "test", oos[f"{m}x"]["sharpe"], "cost stress")
    for (a, b), v in oos_cells.items():
        log_trial("sealed", a, b, 1, "test", v, "grid sensitivity")
    h = oos["1x"]
    print(f"OOS {te[0]}..{te[1]}: Sharpe {h['sharpe']:.2f} (CI {ci[0]:.2f}..{ci[1]:.2f}), ann ret {h['ann_return']:.2%}, vol {h['ann_vol']:.2%}, "
          f"maxDD {h['max_drawdown']:.2%}, trips {h['round_trips']}")
    print(f"DSR prob: {dsr['probability']:.3f}" if dsr else "DSR unavailable")
    print("flags:", flags)
    print("verdict:", decision["verdict"], "| failed:", decision["failed"])
