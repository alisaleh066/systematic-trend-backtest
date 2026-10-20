"""Download the registered v3 series into data/raw_v3 and write data/manifest_v3.json. Computes no returns or statistics."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stb import data, universe_v3 as u3  # noqa: E402

cfg = u3.load_config3()
start, end = cfg["data"]["start"], data.last_complete_month_end_exclusive()
holdout_end = cfg["holdout"][1]
records, dropped = [], {}


def keep(name, series, meta):
    if series.index[0] > __import__("pandas").Timestamp(holdout_end):
        dropped[name] = f"no data before holdout end (first {series.index[0].date()})"
        return
    records.append(u3.write_raw3(name, series, meta))
    print(f"{name}: {meta['source']} {len(series)} rows {series.index[0].date()}..{series.index[-1].date()}")


for t in cfg["universe"]["funds"]:
    try:
        s, rec = data.download_ticker(t, start, end)
        keep(t, s, {"source": rec["source"], "url": rec["url"], "adjustment": rec["adjustment"]})
    except Exception as exc:  # noqa: BLE001 - registered drop rule: log and continue
        dropped[t] = f"download failed: {exc}"
for name, sym in cfg["universe"]["stooq_spot"].items():
    try:
        keep(name, *u3.fetch_stooq(sym, start, end))
    except Exception as exc:  # noqa: BLE001
        dropped[name] = f"download failed: {type(exc).__name__}: {exc}"
for sid in cfg["universe"]["fred_spot"] + list(cfg["universe"]["fred_fx"]) + ["DTB3"]:
    try:
        keep(sid, *u3.fetch_fred(sid, start, end))
    except Exception as exc:  # noqa: BLE001
        dropped[sid] = f"download failed: {type(exc).__name__}: {exc}"

u3.MANIFEST3.write_text(json.dumps({"start": start, "end_exclusive": end, "series": records, "dropped": dropped}, indent=2) + "\n")
print("dropped:", json.dumps(dropped, indent=1))
