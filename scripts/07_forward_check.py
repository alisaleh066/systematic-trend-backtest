"""Prospective paper-trade of the frozen v2 rule (contract line 66). Downloads fresh prices into memory only, computes the
frozen design's month-end target weights dated after the freeze, and writes results/forward_log.json. Claims no performance
until at least one fill has happened."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from stb import data, seal2, universe  # noqa: E402
from stb.engine import simulate  # noqa: E402
from stb.engine_v2 import build_fills_v2  # noqa: E402

cfg = universe.load_config2()
freeze = json.loads(seal2.FREEZE2.read_text())
sel = freeze["selected_design"]
start_after = cfg["registered"]  # the freeze date; only month-end signals after it count
tickers = sel["tradable_assets"]
end = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
px = pd.concat([data.download_ticker(t, cfg["data"]["start"], end)[0] for t in tickers], axis=1)
spy = data.download_ticker("SPY", cfg["data"]["start"], end)[0]
px = px.reindex(spy.index)
first = (pd.Timestamp(start_after) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
fills = build_fills_v2(px, tickers, cfg["signals"][sel["signal"]], sel["vol_window"], cfg["sizing"]["target_vol"], cfg["sizing"]["gross_cap"],
                       first, cfg["universe"]["min_history_sessions"])
log = {"frozen_design": sel["key"], "freeze_date": start_after, "data_through": px.index[-1].strftime("%Y-%m-%d"), "fills": []}
for k, w in fills.items():
    log["fills"].append({"fill_date": px.index[k].strftime("%Y-%m-%d"), "weights": {t: round(float(x), 4) for t, x in zip(tickers, w) if abs(x) > 1e-9}})
if fills:
    rf = pd.Series(0.0, px.index)  # cash leg ignored here: this logs the risky sleeve only
    res = simulate(px.bfill(), rf, fills, cfg["costs"]["commission_bps"] + cfg["costs"]["slippage_bps"], cfg["costs"]["borrow_bps_per_year"])
    begin = px.index[min(fills)]
    log["paper_return_since_first_fill"] = float((1 + res.net.loc[begin:]).prod() - 1)
else:
    log["note"] = "no month-end signal after the freeze date yet; nothing to report"
(ROOT / "results" / "forward_log.json").write_text(json.dumps(log, indent=2) + "\n")
print(json.dumps({k: v for k, v in log.items() if k != "fills"}, indent=2), f"| fills: {len(log['fills'])}")
