"""v3 development report: the frozen v2 rule on 2002-09 onward. Nothing is chosen here, and no holdout-dated statistic is computed."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from stb import pipeline as pl  # noqa: E402
from stb import universe_v3 as u3  # noqa: E402
from stb.engine_v2 import run_design  # noqa: E402
from stb.metrics import window_stats  # noqa: E402
from stb.trials import log_trial  # noqa: E402

cfg = u3.load_config3()
u3.verify_manifest_v3()
panel, rf, log = u3.build_panel3(cfg)
rule = cfg["rule"]
bps = cfg["costs"]["commission_bps"] + cfg["costs"]["slippage_bps"]
dev = [cfg["dev"]["start"], panel.index[-1].strftime("%Y-%m-%d")]
res = run_design(panel, rf, list(panel.columns), cfg["signals"][rule["signal"]], rule["vol_window"], rule["target_vol"], bps,
                 dev[0], cfg["universe"]["min_history_sessions"], rule["gross_cap"], cfg["costs"]["borrow_bps_per_year"])
start = panel.index.searchsorted(pd.Timestamp(dev[0]))
assert res.turnover.iloc[:start].sum() == 0, "positions before the development window"
stats = pl.segment_stats(res, rf, *dev)
crises = {k: window_stats(res.net.loc[a:b], rf) for k, (a, b) in cfg["crises_dev"].items()}
log_trial("dev-v3", 1, rule["target_vol"], 1, "development", stats["sharpe"], "v3 rule (report only)", study="v3")
summary = {"dev_window": dev, "assets": list(panel.columns), "panel_log": log, "development": stats, "crises_dev": crises}
(ROOT / "results" / "v3_dev_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"assets ({len(panel.columns)}): {list(panel.columns)}")
print(f"development {dev[0]}..{dev[1]} (seen before; report only): Sharpe {stats['sharpe']:.2f}, ann ret {stats['ann_return']:.2%}, trips {stats['round_trips']}")
