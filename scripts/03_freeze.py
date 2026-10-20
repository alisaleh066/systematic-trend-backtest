"""Freeze config, engine code and trial count. Needs a clean git tree. Commit freeze.json afterwards."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stb import seal  # noqa: E402
from stb.trials import unique_trial_count  # noqa: E402

dev = json.loads((ROOT / "results" / "dev_summary.json").read_text())
rec = seal.write_freeze(unique_trial_count(), dev["primary"])
print(f"frozen at {rec['git_head'][:8]}, trials={rec['trial_count']}, config={rec['config_sha256'][:12]}, code={rec['code_sha256'][:12]}")
