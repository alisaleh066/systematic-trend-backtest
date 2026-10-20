"""Open the v3 holdout (1988-01-04 to 2002-08-30). Refuses unless freeze-v3.json matches and the tree is clean.
First run writes the seal record; any later run must reproduce identical numbers."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from stb import pipeline as pl  # noqa: E402
from stb import seal3, universe_v3 as u3  # noqa: E402
from stb.dsr import deflated_sharpe  # noqa: E402
from stb.engine import run_benchmark  # noqa: E402
from stb.engine_v2 import run_design  # noqa: E402
from stb.metrics import clip_trades, stationary_bootstrap_sharpe_ci, window_stats  # noqa: E402
from stb.regimes import cell_coverage, decompose, regime_labels  # noqa: E402
from stb.trials import log_trial  # noqa: E402
from stb.verdict import decide, neighbour_gap, overfit_flags  # noqa: E402

RES = ROOT / "results"
OPENED = RES / "v3_seal_opened.json"
freeze = seal3.check_freeze3()
reproducing = OPENED.exists()
if not reproducing:
    OPENED.write_text(json.dumps({"opened_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  **{k: freeze[k] for k in ("git_head", "config_sha256", "code_sha256")}}, indent=2) + "\n")

cfg = u3.load_config3()
u3.verify_manifest_v3()
full, rf_full, log = u3.build_panel3(cfg)
ho = cfg["holdout"]
panel, rf = full.loc[: ho[1]], rf_full.loc[: ho[1]]  # everything after the holdout end is dropped before computing
assets = list(panel.columns)
rule, costs = cfg["rule"], cfg["costs"]
bps, minh = costs["commission_bps"] + costs["slippage_bps"], cfg["universe"]["min_history_sessions"]


def run(sig, vol, mult=1):
    return run_design(panel, rf, assets, cfg["signals"][sig], rule["vol_window"], vol, bps * mult, ho[0], minh, rule["gross_cap"], costs["borrow_bps_per_year"])


runs = {m: run(rule["signal"], rule["target_vol"], m) for m in [1, *costs["stress_multipliers"]]}
oos = {f"{m}x": pl.segment_stats(r, rf, *ho) for m, r in runs.items()}
primary = runs[1]
ret = primary.net.loc[ho[0]:ho[1]]
excess = (ret - rf.loc[ret.index]).to_numpy()
ci = stationary_bootstrap_sharpe_ci(excess, cfg["bootstrap"]["mean_block"], cfg["bootstrap"]["resamples"], cfg["bootstrap"]["seed"])
v1dev = json.loads((RES / "dev_summary.json").read_text())
v2dev = json.loads((RES / "v2_dev_summary.json").read_text())
dsr = deflated_sharpe(excess, v1dev["trial_daily_sharpes_dev"] + v2dev["trial_daily_sharpes_v2"], freeze["trial_count"])

sigs = list(cfg["signals"])
cells = {(k + 1, v): pl.segment_stats(run(s, v), rf, *ho)["sharpe"] for k, s in enumerate(sigs) for v in cfg["sensitivity_vol_targets"]}

pxb = panel[["VFINX", "VBMFX"]].bfill()
bench, bench_ret = {}, {}
for name, w in cfg["benchmarks"].items():
    b = run_benchmark(pxb, rf, w, ho[0], bps, monthly_rebalance=(name == "sixty_forty"))
    bench_ret[name] = b.net.loc[ho[0]:ho[1]]
    bench[name] = window_stats(bench_ret[name], rf, turnover=b.turnover.loc[ho[0]:ho[1]])


def crisis_rows(windows, res, px, rates):
    out = {}
    for name, (a, b) in windows.items():
        out[name] = {"window": [a, b], "strategy": window_stats(res.net.loc[a:b], rates)}
        for bn, w in cfg["benchmarks"].items():
            r = run_benchmark(px, rates, w, a, bps, monthly_rebalance=(bn == "sixty_forty"))
            out[name][bn] = window_stats(r.net.loc[a:b], rates)
    return out


crises_ho = crisis_rows(cfg["crises_holdout"], primary, pxb, rf)
dev_res = run_design(full, rf_full, assets, cfg["signals"][rule["signal"]], rule["vol_window"], rule["target_vol"], bps, cfg["dev"]["start"], minh, rule["gross_cap"], costs["borrow_bps_per_year"])
crises_dev = crisis_rows(cfg["crises_dev"], dev_res, full[["VFINX", "VBMFX"]].bfill(), rf_full)

labels = regime_labels(panel["VFINX"], cfg["regimes"]["er_threshold"], cfg["regimes"]["er_window"], cfg["regimes"]["vol_window"], cfg["regimes"]["vol_median_lookback"])
oos_trades = clip_trades(primary.trades, *ho)
coverage = cell_coverage(labels, oos_trades, cfg["regimes"]["min_trades_per_cell"])
decomp = decompose(ret, labels)

t0, t1 = pd.Timestamp(ho[0]), pd.Timestamp(ho[1])
full_years = [y for y in range(t0.year, t1.year + 1) if pd.Timestamp(y, 1, 1) >= t0 - pd.Timedelta(days=3) and pd.Timestamp(y, 12, 31) <= t1]
# the rule was chosen in v2, so v2's selection diagnostics are inherited, not recomputed
v2sel = v2dev["designs"][v2dev["selected"]["key"]]["validation_sharpe"]
others = sorted(v["validation_sharpe"] for k, v in v2dev["designs"].items() if k != v2dev["selected"]["key"])
gap_val = float(v2sel - pd.Series(others).median())
gap_oos = neighbour_gap(cells, (1, rule["target_vol"]), [1, 2, 3], cfg["sensitivity_vol_targets"])
flags = overfit_flags(cfg["flags"], oos["1x"]["sharpe"], dsr, v2dev["positive_validation_share"], gap_val, gap_oos, oos["1x"]["round_trips"])
decision = decide(oos["1x"], oos["2x"], ci[0], len(full_years), coverage["sparse"], cfg["flags"]["min_oos_trades"])

results = {
    "version": "v3: frozen v2 rule, long-history proxies, pre-sample holdout (departs from contract line 60: not the final 20%)",
    "holdout_window": ho, "assets": assets, "panel_log": log,
    "headline_holdout_1x_costs": oos["1x"], "holdout_cost_stress": oos,
    "sharpe_bootstrap_95ci": list(ci), "deflated_sharpe": dsr, "benchmarks_holdout": bench,
    "sensitivity_sharpe": {f"{k}|{v}": s for (k, v), s in cells.items()},
    "neighbour_gap": {"inherited_v2_validation": gap_val, "holdout_heatmap": gap_oos},
    "crisis_periods_holdout": crises_ho, "crisis_periods_development": crises_dev,
    "regime_coverage": coverage, "regime_decomposition_holdout": decomp,
    "complete_holdout_years": full_years, "trial_count": freeze["trial_count"], "flags": flags, "verdict": decision,
}

if reproducing:
    prior = json.loads((RES / "v3_results.json").read_text())
    if json.dumps(prior, sort_keys=True) != json.dumps(json.loads(json.dumps(results)), sort_keys=True):
        sys.exit("REPRODUCTION FAILED: numbers differ from the first holdout run")
    print("reproduced identical results")
else:
    (RES / "v3_results.json").write_text(json.dumps(results, indent=2) + "\n")
    pd.DataFrame({"strategy": ret, **bench_ret, "rf": rf.loc[ret.index]}).to_csv(RES / "v3_oos_daily.csv", index_label="date")
    oos_trades.to_csv(RES / "v3_oos_trades.csv", index=False)
    for m in costs["stress_multipliers"]:
        log_trial("holdout-v3", 1, rule["target_vol"], m, "holdout", oos[f"{m}x"]["sharpe"], "v3 cost stress", study="v3")
    for (k, v), s in cells.items():
        log_trial("holdout-v3", k, v, 1, "holdout", s, "v3 sensitivity", study="v3")
    h = oos["1x"]
    print(f"HOLDOUT {ho[0]}..{ho[1]}: Sharpe {h['sharpe']:.2f} (CI {ci[0]:.2f}..{ci[1]:.2f}), ann ret {h['ann_return']:.2%}, vol {h['ann_vol']:.2%}, "
          f"maxDD {h['max_drawdown']:.2%}, trips {h['round_trips']}")
    print(f"DSR prob: {dsr['probability']:.3f}" if dsr else "DSR unavailable")
    print("flags:", flags)
    print("verdict:", decision["verdict"], "| failed:", decision["failed"])
