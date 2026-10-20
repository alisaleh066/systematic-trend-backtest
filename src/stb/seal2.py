"""v2 test lock: the holdout needs a freeze-v2.json that still matches config and code, and a clean git tree."""
from __future__ import annotations

import json
from pathlib import Path

from stb import seal
from stb.seal import SealError, git, tree_clean

ROOT = seal.ROOT
FREEZE2 = ROOT / "freeze-v2.json"
CONFIG2 = ROOT / "config" / "v2.json"
V2_MODULES = ["universe", "signals_v2", "engine_v2", "seal2"]


def config2_hash() -> str:
    return seal._sha([CONFIG2])


def code2_hash() -> str:
    # engine.simulate, metrics, dsr, verdict, regimes and signals are reused, so the whole v1 module set is covered too
    return seal._sha([seal.SRC / f"{m}.py" for m in seal.FROZEN_MODULES + V2_MODULES])


def write_freeze2(selected: dict, trial_count: int, dev_summary: dict, clean=tree_clean, path: Path = FREEZE2) -> dict:
    if not clean():
        raise SealError("git tree is dirty: commit everything before freezing")
    rec = {"config_sha256": config2_hash(), "code_sha256": code2_hash(), "git_head": git("rev-parse", "HEAD"),
           "selected_design": selected, "trial_count": trial_count, "dev_summary": dev_summary}
    path.write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def check_freeze2(clean=tree_clean, path: Path = FREEZE2) -> dict:
    if not path.exists():
        raise SealError("no freeze-v2.json: run 10_freeze_v2.py first")
    rec = json.loads(path.read_text())
    if rec["config_sha256"] != config2_hash():
        raise SealError("v2 config changed since the freeze: that is a new version, not a rerun")
    if rec["code_sha256"] != code2_hash():
        raise SealError("v2 engine code changed since the freeze: that is a new version, not a rerun")
    if not clean():
        raise SealError("git tree is dirty: commit before opening the holdout")
    return rec
