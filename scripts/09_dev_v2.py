"""v2 development run: 12 registered designs on 2008-07 onward (train/validation). Nothing is evaluated on holdout dates:
positions are zero before the first dev fill, and only dev-window statistics are computed."""
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from stb import pipeline as pl  # noqa: E402
from stb import universe  # noqa: E402
from stb.engine import run_benchmark  # noqa: E402
from stb.engine_v2 import run_design, split_dev  # noqa: E402
from stb.metrics import window_stats  # noqa: E402
from stb.trials import log_trial  # noqa: E402

cfg = universe.load_config2()
universe.verify_manifest_v2()
u16, extra = cfg["universe"]["u16"], cfg["universe"]["u24_extra"]
panel, log = universe.build_panel(u16 + extra, cfg["universe"]["max_gap_ffill"], cfg["universe"]["max_total_gap_days"])
rf = universe.cash_returns(panel.index)
sets = {"u16": [t for t in u16 if t in panel], "u24": [t for t in u16 + extra if t in panel]}

dev_start = cfg["dev"]["start"]
split = split_dev(panel.index, dev_start, cfg["dev"]["train_frac"])
tr, va = split["train"], split["validation"]
bps = cfg["costs"]["commission_bps"] + cfg["costs"]["slippage_bps"]
tv, gcap, borrow = cfg["sizing"]["target_vol"], cfg["sizing"]["gross_cap"], cfg["costs"]["borrow_bps_per_year"]
minh = cfg["universe"]["min_history_sessions"]

designs, dev_sharpes = {}, []
for k, (sig, uni, vw) in enumerate(itertools.product(cfg["designs"]["signal"], cfg["designs"]["universe"], cfg["designs"]["vol_window"])):
    key = f"{sig}|{uni}|{vw}"
    res = run_design(panel, rf, sets[uni], cfg["signals"][sig], vw, tv, bps, dev_start, minh, gcap, borrow)
    s_tr, s_va = pl.segment_stats(res, rf, *tr), pl.segment_stats(res, rf, *va)
    d = pl.excess_daily_sharpe(res, rf, tr[0], va[1])
    dev_sharpes.append(d)
    designs[key] = {"train_sharpe": s_tr["sharpe"], "validation_sharpe": s_va["sharpe"], "dev_daily_sharpe": d,
                    "train": s_tr, "validation": s_va}
    sig_id = cfg["designs"]["signal"].index(sig) + 1
    log_trial("dev-v2", sig_id, tv, 1, "train", s_tr["sharpe"], key, study="v2")
    log_trial("dev-v2", sig_id, tv, 1, "validation", s_va["sharpe"], key, study="v2")
    print(f"{key:12s} train {s_tr['sharpe']:+.2f}  validation {s_va['sharpe']:+.2f}  val trips {s_va['round_trips']}")

selected = max(designs, key=lambda k: designs[k]["validation_sharpe"])
sig, uni, vw = selected.split("|")
vw = int(vw)
stress = {}
for m in cfg["costs"]["stress_multipliers"]:
    res = run_design(panel, rf, sets[uni], cfg["signals"][sig], vw, tv, bps * m, dev_start, minh, gcap, borrow)
    stress[f"{m}x"] = pl.segment_stats(res, rf, *va)
    log_trial("dev-v2", cfg["designs"]["signal"].index(sig) + 1, tv, m, "validation", stress[f"{m}x"]["sharpe"], f"{selected} cost stress", study="v2")

px16 = panel[sets["u16"]].bfill()
bench = {}
for name, w in cfg["benchmarks"].items():
    b = run_benchmark(px16, rf, w, va[0], bps, monthly_rebalance=(name == "sixty_forty"))
    bench[name] = window_stats(b.net.loc[va[0]:va[1]], rf)

v1 = json.loads((ROOT / "results" / "dev_summary.json").read_text())
summary = {
    "split": split,
    "designs": designs,
    "selected": {"key": selected, "signal": sig, "universe": uni, "vol_window": vw, "tradable_assets": sets[uni]},
    "positive_validation_share": float(np.mean([v["validation_sharpe"] > 0 for v in designs.values()])),
    "trial_daily_sharpes_v2": dev_sharpes,
    "trial_daily_sharpes_v1": v1["trial_daily_sharpes_dev"],
    "cost_stress_validation_selected": stress,
    "benchmarks_validation": bench,
    "panel_log": log,
}
(ROOT / "results" / "v2_dev_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
s = designs[selected]
print(f"\nselected {selected}: train Sharpe {s['train_sharpe']:+.2f}, validation Sharpe {s['validation_sharpe']:+.2f}; "
      f"positive-validation designs {summary['positive_validation_share']:.0%}")
print("dev window", split["train"], split["validation"])
