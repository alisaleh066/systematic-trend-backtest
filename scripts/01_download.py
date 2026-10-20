"""Download all tickers, write data/raw/*.csv and data/manifest.json. No returns are computed here."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stb import data  # noqa: E402

cfg = json.loads((ROOT / "config" / "primary.json").read_text())
tickers = cfg["universe"] + [cfg["cash"]]
start = cfg["data"]["start"]
end = data.last_complete_month_end_exclusive()

records = []
for t in tickers:
    series, rec = data.download_ticker(t, start, end)
    records.append(data.write_raw(series, rec))
    print(f"{t}: {rec['source']} {rec['rows']} rows {rec['first']}..{rec['last']}")
data.write_manifest(records, start, end)
print(f"manifest written, end_exclusive={end}")
