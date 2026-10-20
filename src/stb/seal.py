"""Test-set lock: the sealed run needs a freeze record that still matches config and code, and a clean git tree."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "freeze.json"
CONFIG = ROOT / "config" / "primary.json"
SRC = ROOT / "src" / "stb"
FROZEN_MODULES = ["data", "signals", "portfolio", "engine", "metrics", "dsr", "split", "pipeline", "regimes", "verdict", "trials"]


class SealError(RuntimeError):
    pass


def _sha(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode() + p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def config_hash() -> str:
    return _sha([CONFIG])


def code_hash() -> str:
    return _sha([SRC / f"{m}.py" for m in FROZEN_MODULES])


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def tree_clean() -> bool:
    return git("status", "--porcelain") == ""


def write_freeze(trial_count: int, validation_summary: dict, clean=tree_clean, path: Path = FREEZE) -> dict:
    if not clean():
        raise SealError("git tree is dirty: commit everything before freezing")
    rec = {
        "config_sha256": config_hash(),
        "code_sha256": code_hash(),
        "git_head": git("rev-parse", "HEAD"),
        "trial_count": trial_count,
        "validation_primary": validation_summary,
    }
    path.write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def check_freeze(clean=tree_clean, path: Path = FREEZE) -> dict:
    if not path.exists():
        raise SealError("no freeze.json: run 03_freeze.py first")
    rec = json.loads(path.read_text())
    if rec["config_sha256"] != config_hash():
        raise SealError("config changed since the freeze: that is a new version, not a rerun")
    if rec["code_sha256"] != code_hash():
        raise SealError("engine code changed since the freeze: that is a new version, not a rerun")
    if not clean():
        raise SealError("git tree is dirty: commit before opening the sealed test")
    return rec
