"""Secondary study: do the registered radar break rules predict next-month returns? Reported separately."""
import csv
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stb import pipeline as pl  # noqa: E402
from stb.data import _from_yfinance, last_complete_month_end_exclusive  # noqa: E402
from stb.radar import benjamini_hochberg, break_vs_hold, classify_rules  # noqa: E402
from stb.trials import log_trial  # noqa: E402

RAW = ROOT / "data" / "raw_radar"
RAW.mkdir(parents=True, exist_ok=True)
YAHOO = ["SPY", "TLT", "RSP", "QQQ", "SOXX", "^VIX", "BZ=F", "XLE", "ZW=F", "DX-Y.NYB", "GC=F"]
FRED = ["DFII10", "BAMLH0A0HYM2"]
DIFFERENCE = set(FRED)
# (left, right, expected, ratio_confirmation): the registered pairs, from newsletter-config.json radar_groups
PAIRS = [("SPY", "TLT", -1, False), ("RSP", "SPY", 1, True), ("QQQ", "SPY", 1, True), ("SOXX", "SPY", 1, True),
         ("SPY", "^VIX", -1, False), ("BZ=F", "SPY", -1, False), ("BZ=F", "XLE", 1, False), ("BZ=F", "^VIX", 1, False),
         ("ZW=F", "DX-Y.NYB", -1, False), ("GC=F", "DFII10", -1, False), ("BAMLH0A0HYM2", "SPY", -1, False)]
START, END = "2006-01-01", last_complete_month_end_exclusive()
now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: E731

series, manifest = {}, []
for s in YAHOO:
    ser = _from_yfinance(s, START, END)
    ser.index = pd.to_datetime(ser.index).tz_localize(None)
    series[s] = ser
    manifest.append({"symbol": s, "source": "yfinance", "downloaded_utc": now()})
for s in FRED:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}&cosd={START}"
    rows = list(csv.DictReader(io.StringIO(urlopen(url, timeout=60).read().decode("utf-8-sig"))))
    key = "observation_date" if "observation_date" in rows[0] else "DATE"
    vals = {pd.Timestamp(r[key]): float(r[s]) for r in rows if r[s] not in ("", ".") and r[key] < END}
    series[s] = pd.Series(vals).sort_index().rename(s)
    manifest.append({"symbol": s, "source": "FRED", "url": url, "downloaded_utc": now()})
for rec in manifest:
    ser = series[rec["symbol"]]
    path = RAW / f"{rec['symbol'].replace('^', '_').replace('=', '_')}.csv"
    ser.to_csv(path, header=["value"], index_label="date")
    rec.update(rows=int(len(ser)), first=ser.index[0].strftime("%Y-%m-%d"), last=ser.index[-1].strftime("%Y-%m-%d"), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(ROOT / "data" / "manifest_radar.json").write_text(json.dumps({"start": START, "end_exclusive": END, "series": manifest}, indent=2) + "\n")

cfg, split = pl.load_config(), pl.load_split()
spy = series["SPY"]
cal = spy.index[spy.index >= pd.Timestamp("2006-01-01")]
ym = cal.year * 100 + cal.month
month_ends = [k for k in range(len(cal)) if k == len(cal) - 1 or ym[k] != ym[k + 1]]
eval_start, train_end, val_end = (pd.Timestamp(split["eval_start"]), pd.Timestamp(split["train"][1]), pd.Timestamp(split["validation"][1]))

rows, pair_info = [], {}
for left, right, expected, ratio_conf in PAIRS:
    both = pd.concat([series[left], series[right]], axis=1, join="inner").dropna()
    both.columns = ["x", "y"]
    pair_info[f"{left}/{right}"] = {"aligned_rows": int(len(both)), "first": both.index[0].strftime("%Y-%m-%d"), "reads": 0}
    pos = {d: k for k, d in enumerate(both.index)}
    for a, b in zip(month_ends, month_ends[1:]):
        d = cal[a]
        if d < eval_start or d not in pos or b + 1 >= len(cal):
            continue
        k = pos[d]
        res = classify_rules(both["x"].iloc[: k + 1].tolist(), both["y"].iloc[: k + 1].tolist(), expected, left in DIFFERENCE, right in DIFFERENCE, ratio_conf)
        if res is None:
            continue
        pair_info[f"{left}/{right}"]["reads"] += 1
        t0, t1 = cal[a + 1], cal[b + 1]  # fill at the close after the signal; hold to the next month's fill
        window = spy.loc[t0:t1]
        left_ret = float(series[left].loc[t1] / series[left].loc[t0] - 1) if left not in DIFFERENCE and t0 in series[left].index and t1 in series[left].index else np.nan
        rows.append({"pair": f"{left}/{right}", "month": d.year * 100 + d.month, "segment": "train" if d <= train_end else "validation" if d <= val_end else "test",
                     "spy_ret": float(spy.loc[t1] / spy.loc[t0] - 1), "left_ret": left_ret,
                     "spy_vol": float(window.pct_change().dropna().std() * np.sqrt(252)), **res})
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "radar_observations.csv", index=False)

RULES, OUTCOMES = ["divergence", "correlation", "zscore", "any_break"], ["spy_ret", "left_ret", "spy_vol"]
out = {"pairs": pair_info, "segments": {}, "headline_segment": "test"}
for seg in ("train", "validation", "test"):
    sub = df[df["segment"] == seg]
    tests = {}
    for r in RULES:
        for o in OUTCOMES:
            tests[f"{r}|{o}"] = break_vs_hold(sub["month"].to_numpy(), sub[r].to_numpy(bool), sub[o].to_numpy(float), resamples=cfg["bootstrap"]["resamples"], seed=cfg["bootstrap"]["seed"])
    keys = [k for k, v in tests.items() if v]
    keep = benjamini_hochberg([tests[k]["p_value"] for k in keys])
    for k, passed in zip(keys, keep):
        tests[k]["bh_significant_5pct"] = bool(passed)
    out["segments"][seg] = {"observations": int(len(sub)), "tests": tests}
    if seg == "test":
        for k, v in tests.items():
            log_trial("radar", 0, 0, 1, "test", float("nan"), f"{k} diff={v['diff']:.5f}" if v else f"{k} too few", study="radar")
(ROOT / "results" / "radar_results.json").write_text(json.dumps(out, indent=2) + "\n")

t = out["segments"]["test"]
print(f"observations: " + ", ".join(f"{s}={out['segments'][s]['observations']}" for s in out["segments"]))
for k, v in t["tests"].items():
    print(f"{k:24s}", "too few" if not v else f"diff {v['diff']:+.4f} CI [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] n={v['n_break']}/{v['n_hold']} BH={v['bh_significant_5pct']}")
print({k: v["reads"] for k, v in pair_info.items()})
