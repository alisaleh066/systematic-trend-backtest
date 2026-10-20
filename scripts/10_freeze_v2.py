"""Freeze v2: tests must be green and the tree clean. Records config hash, code hash, selected design and trial count."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stb import seal2, universe  # noqa: E402

tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
print(tests.stdout[-400:])
if tests.returncode != 0:
    sys.exit("tests are not green: not freezing")

cfg = universe.load_config2()
dev = json.loads((ROOT / "results" / "v2_dev_summary.json").read_text())
sel = dev["selected"]
rec = seal2.write_freeze2(sel, cfg["trial_count"], {"selected": dev["designs"][sel["key"]], "positive_validation_share": dev["positive_validation_share"]})
print(f"frozen v2 at {rec['git_head'][:8]}: {sel['key']}, trials={rec['trial_count']}, config={rec['config_sha256'][:12]}, code={rec['code_sha256'][:12]}")
