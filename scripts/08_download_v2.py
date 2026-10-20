"""Download v2 data (U24 ETFs from 1999, FRED DTB3) into data/raw_v2 and write data/manifest_v2.json. Computes no returns."""
import csv
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from stb import data, universe  # noqa: E402

cfg = universe.load_config2()
tickers = cfg["universe"]["u16"] + cfg["universe"]["u24_extra"]
start, end = cfg["data"]["start"], data.last_complete_month_end_exclusive()

records = []
for t in tickers:
    series, rec = data.download_ticker(t, start, end)
    rec["file"] = t
    records.append(universe.write_raw_v2(series, rec))
    print(f"{t}: {rec['source']} {rec['rows']} rows {rec['first']}..{rec['last']}")

url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3&cosd={start}"
rows = list(csv.DictReader(io.StringIO(urlopen(url, timeout=60).read().decode("utf-8-sig"))))
key = "observation_date" if "observation_date" in rows[0] else "DATE"
vals = {pd.Timestamp(r[key]): float(r["DTB3"]) for r in rows if r["DTB3"] not in ("", ".") and r[key] < end}
dtb = pd.Series(vals).sort_index()
rec = {"ticker": "DTB3", "file": "DTB3", "source": "FRED", "url": url, "adjustment": "annualised % yield",
       "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rows": len(dtb),
       "first": dtb.index[0].strftime("%Y-%m-%d"), "last": dtb.index[-1].strftime("%Y-%m-%d"), "fallback_errors": []}
records.append(universe.write_raw_v2(dtb, rec))
print(f"DTB3: FRED {rec['rows']} rows {rec['first']}..{rec['last']}")

panel, log = universe.build_panel(tickers, cfg["universe"]["max_gap_ffill"], cfg["universe"]["max_total_gap_days"])
(universe.MANIFEST2).write_text(json.dumps({"start": start, "end_exclusive": end, "series": records, "panel_log": log,
                                            "first_listing": {t: panel[t].first_valid_index().strftime("%Y-%m-%d") for t in panel}}, indent=2) + "\n")
print("excluded:", log["excluded"], "| gap days:", {k: v for k, v in log["gap_days"].items() if v})
print("first listings:", {t: panel[t].first_valid_index().strftime("%Y-%m-%d") for t in panel})
