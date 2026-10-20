"""Open the v2 pre-sample holdout (2002-09-01 to 2008-06-30). Refuses unless freeze-v2.json matches and the tree is clean.
First run writes the seal record; any later run must reproduce identical numbers."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stb import pipeline as pl  # noqa: E402
from stb import seal2, universe  # noqa: E402
from stb.dsr import deflated_sharpe  # noqa: E402
from stb.engine import run_benchmark  # noqa: E402
from stb.engine_v2 import run_design  # noqa: E402
from stb.metrics import clip_trades, stationary_bootstrap_sharpe_ci, window_stats  # noqa: E402
from stb.regimes import cell_coverage, decompose, regime_labels  # noqa: E402
from stb.trials import log_trial  # noqa: E402
from stb.verdict import decide, neighbour_gap, overfit_flags  # noqa: E402

RES = ROOT / "results"
OPENED = RES / "v2_seal_opened.json"
freeze = seal2.check_freeze2()
reproducing = OPENED.exists()
if not reproducing:
    OPENED.write_text(json.dumps({"opened_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  **{k: freeze[k] for k in ("git_head", "config_sha256", "code_sha256")}}, indent=2) + "\n")

cfg = universe.load_config2()
dev = json.loads((RES / "v2_dev_summary.json").read_text())
universe.verify_manifest_v2()
u16, extra = cfg["universe"]["u16"], cfg["universe"]["u24_extra"]
full, _ = universe.build_panel(u16 + extra, cfg["universe"]["max_gap_ffill"], cfg["universe"]["max_total_gap_days"])
ho = cfg["holdout"]
panel = full.loc[: ho[1]]  # everything after the holdout end is dropped before anything is computed
rf = universe.cash_returns(full.index).loc[: ho[1]]
sets = {"u16": [t for t in u16 if t in panel], "u24": [t for t in u16 + extra if t in panel]}
sel = freeze["selected_design"]
sig, uni, vw = sel["signal"], sel["universe"], sel["vol_window"]
bps, tv, gcap = cfg["costs"]["commission_bps"] + cfg["costs"]["slippage_bps"], cfg["sizing"]["target_vol"], cfg["sizing"]["gross_cap"]
borrow, minh = cfg["costs"]["borrow_bps_per_year"], cfg["universe"]["min_history_sessions"]


def run(sig_name, universe_name, window, vol, mult=1):
    return run_design(panel, rf, sets[universe_name], cfg["signals"][sig_name], window, vol, bps * mult, ho[0], minh, gcap, borrow)


runs = {m: run(sig, uni, vw, tv, m) for m in [1, *cfg["costs"]["stress_multipliers"]]}
oos = {f"{m}x": pl.segment_stats(r, rf, *ho) for m, r in runs.items()}
primary = runs[1]
ret = primary.net.loc[ho[0]:ho[1]]
excess = (ret - rf.loc[ret.index]).to_numpy()
ci = stationary_bootstrap_sharpe_ci(excess, cfg["bootstrap"]["mean_block"], cfg["bootstrap"]["resamples"], cfg["bootstrap"]["seed"])
trial_sharpes = dev["trial_daily_sharpes_v2"] + dev["trial_daily_sharpes_v1"]
dsr = deflated_sharpe(excess, trial_sharpes, freeze["trial_count"])

# sensitivity: signal variant x vol target on the selected universe and vol window
sigs = cfg["designs"]["signal"]
cells = {(k + 1, v): pl.segment_stats(run(s, uni, vw, v), rf, *ho)["sharpe"] for k, s in enumerate(sigs) for v in cfg["sensitivity_vol_targets"]}
# v1's primary rule (S1, U16, 63) on the same holdout
v1_rule = pl.segment_stats(run("S1", "u16", 63, tv), rf, *ho)

px16 = panel[["SPY", "IEF"]].bfill()
bench, bench_ret = {}, {}
for name, w in cfg["benchmarks"].items():
    b = run_benchmark(px16, rf, w, ho[0], bps, monthly_rebalance=(name == "sixty_forty"))
    bench_ret[name] = b.net.loc[ho[0]:ho[1]]
    bench[name] = window_stats(bench_ret[name], rf, turnover=b.turnover.loc[ho[0]:ho[1]])
    bench[name]["costs_paid"] = float(b.cost.loc[ho[0]:ho[1]].sum())

# development-window crises for the selected design (development data, not a test)
dev_run = run_design(full, universe.cash_returns(full.index), sets[uni], cfg["signals"][sig], vw, tv, bps, cfg["dev"]["start"], minh, gcap, borrow)
rf_full = universe.cash_returns(full.index)
crises = {}
for name, (a, b) in cfg["crises"].items():
    crises[name] = {"segment": "development", "window": [a, b], "strategy": window_stats(dev_run.net.loc[a:b], rf_full)}
    for bn in ("spy", "sixty_forty"):
        px = full[["SPY", "IEF"]].bfill()
        r = run_benchmark(px, rf_full, cfg["benchmarks"][bn], a, bps, monthly_rebalance=(bn == "sixty_forty"))
        crises[name][bn] = window_stats(r.net.loc[a:b], rf_full)

labels = regime_labels(panel["SPY"], cfg["regimes"]["er_threshold"], cfg["regimes"]["er_window"], cfg["regimes"]["vol_window"], cfg["regimes"]["vol_median_lookback"])
oos_trades = clip_trades(primary.trades, *ho)
coverage = cell_coverage(labels, oos_trades, cfg["regimes"]["min_trades_per_cell"])
decomp = decompose(ret, labels)

t0, t1 = pd.Timestamp(ho[0]), pd.Timestamp(ho[1])
full_years = [y for y in range(t0.year, t1.year + 1) if pd.Timestamp(y, 1, 1) >= t0 and pd.Timestamp(y, 12, 31) <= t1]
others = [v["validation_sharpe"] for k, v in dev["designs"].items() if k != sel["key"]]
gap_val = float(dev["designs"][sel["key"]]["validation_sharpe"] - np.median(others))
gap_oos = neighbour_gap(cells, (sigs.index(sig) + 1, tv), [1, 2, 3], cfg["sensitivity_vol_targets"])
flags = overfit_flags(cfg["flags"], oos["1x"]["sharpe"], dsr, dev["positive_validation_share"], gap_val, gap_oos, oos["1x"]["round_trips"])
decision = decide(oos["1x"], oos["2x"], ci[0], len(full_years), coverage["sparse"], cfg["flags"]["min_oos_trades"])

results = {
    "version": "v2 pre-sample holdout (departs from contract line 60: not the final 20%)",
    "selected_design": sel["key"], "holdout_window": ho,
    "headline_holdout_1x_costs": oos["1x"], "holdout_cost_stress": oos,
    "sharpe_bootstrap_95ci": list(ci), "deflated_sharpe": dsr, "benchmarks_holdout": bench,
    "v1_primary_rule_on_holdout": v1_rule,
    "sensitivity_sharpe": {f"{k}|{v}": s for (k, v), s in cells.items()},
    "neighbour_gap": {"selected_vs_other_designs_validation": gap_val, "holdout_heatmap": gap_oos},
    "crisis_periods_development": crises, "regime_coverage": coverage, "regime_decomposition_holdout": decomp,
    "complete_holdout_years": full_years, "trial_count": freeze["trial_count"], "flags": flags, "verdict": decision,
}

if reproducing:
    prior = json.loads((RES / "v2_results.json").read_text())
    if json.dumps(prior, sort_keys=True) != json.dumps(json.loads(json.dumps(results)), sort_keys=True):
        sys.exit("REPRODUCTION FAILED: numbers differ from the first holdout run")
    print("reproduced identical results")
else:
    (RES / "v2_results.json").write_text(json.dumps(results, indent=2) + "\n")
    pd.DataFrame({"strategy": ret, **bench_ret, "rf": rf.loc[ret.index]}).to_csv(RES / "v2_oos_daily.csv", index_label="date")
    oos_trades.to_csv(RES / "v2_oos_trades.csv", index=False)
    k1 = sigs.index(sig) + 1
    for m in cfg["costs"]["stress_multipliers"]:
        log_trial("holdout-v2", k1, tv, m, "holdout", oos[f"{m}x"]["sharpe"], f"{sel['key']} cost stress", study="v2")
    for (k, v), s in cells.items():
        log_trial("holdout-v2", k, v, 1, "holdout", s, f"{sel['key']} sensitivity", study="v2")
    h = oos["1x"]
    print(f"HOLDOUT {ho[0]}..{ho[1]}: Sharpe {h['sharpe']:.2f} (CI {ci[0]:.2f}..{ci[1]:.2f}), ann ret {h['ann_return']:.2%}, vol {h['ann_vol']:.2%}, "
          f"maxDD {h['max_drawdown']:.2%}, trips {h['round_trips']}")
    print(f"DSR prob: {dsr['probability']:.3f}" if dsr else "DSR unavailable")
    print("flags:", flags)
    print("verdict:", decision["verdict"], "| failed:", decision["failed"])
