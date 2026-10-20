"""Freeze v3: tests must be green and the tree clean. Records config hash, code hash and trial count."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stb import seal3, universe_v3 as u3  # noqa: E402

tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
print(tests.stdout[-300:])
if tests.returncode != 0:
    sys.exit("tests are not green: not freezing")
cfg = u3.load_config3()
dev = json.loads((ROOT / "results" / "v3_dev_summary.json").read_text())
rec = seal3.write_freeze3(cfg["trial_count"], {"development_sharpe": dev["development"]["sharpe"], "assets": dev["assets"]})
print(f"frozen v3 at {rec['git_head'][:8]}, trials={rec['trial_count']}, config={rec['config_sha256'][:12]}, code={rec['code_sha256'][:12]}")
