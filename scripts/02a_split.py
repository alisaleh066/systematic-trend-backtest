"""Write config/split.json from the session calendar. Only index dates are read, no prices."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from stb.data import RAW  # noqa: E402
from stb.split import compute_split  # noqa: E402

cfg = json.loads((ROOT / "config" / "primary.json").read_text())
tickers = cfg["universe"] + [cfg["cash"]]
index = None
for t in tickers:
    idx = pd.read_csv(RAW / f"{t}.csv", usecols=["date"], parse_dates=["date"])["date"]
    index = pd.DatetimeIndex(idx) if index is None else index.intersection(pd.DatetimeIndex(idx))

split = compute_split(index, cfg["data"]["eval_start"], cfg["split"]["train"], cfg["split"]["validation"])
(ROOT / "config" / "split.json").write_text(json.dumps(split, indent=2) + "\n")
print(json.dumps(split, indent=2))
