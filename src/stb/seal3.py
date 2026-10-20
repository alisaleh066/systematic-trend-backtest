"""v3 test lock: the holdout needs a freeze-v3.json that still matches config and code, and a clean git tree."""
from __future__ import annotations

import json
from pathlib import Path

from stb import seal, seal2
from stb.seal import SealError, git, tree_clean

FREEZE3 = seal.ROOT / "freeze-v3.json"
CONFIG3 = seal.ROOT / "config" / "v3.json"
V3_MODULES = ["universe_v3", "seal3"]


def config3_hash() -> str:
    return seal._sha([CONFIG3])


def code3_hash() -> str:
    return seal._sha([seal.SRC / f"{m}.py" for m in seal.FROZEN_MODULES + seal2.V2_MODULES + V3_MODULES])


def write_freeze3(trial_count: int, dev_summary: dict, clean=tree_clean, path: Path = FREEZE3) -> dict:
    if not clean():
        raise SealError("git tree is dirty: commit everything before freezing")
    rec = {"config_sha256": config3_hash(), "code_sha256": code3_hash(), "git_head": git("rev-parse", "HEAD"),
           "trial_count": trial_count, "dev_summary": dev_summary}
    path.write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def check_freeze3(clean=tree_clean, path: Path = FREEZE3) -> dict:
    if not path.exists():
        raise SealError("no freeze-v3.json: run 15_freeze_v3.py first")
    rec = json.loads(path.read_text())
    if rec["config_sha256"] != config3_hash():
        raise SealError("v3 config changed since the freeze: that is a new version, not a rerun")
    if rec["code_sha256"] != code3_hash():
        raise SealError("v3 engine code changed since the freeze: that is a new version, not a rerun")
    if not clean():
        raise SealError("git tree is dirty: commit before opening the holdout")
    return rec
