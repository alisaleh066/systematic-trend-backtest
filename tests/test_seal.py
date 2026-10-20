import json

import pytest

from stb import seal


def test_refuses_without_freeze(tmp_path):
    with pytest.raises(seal.SealError, match="no freeze"):
        seal.check_freeze(clean=lambda: True, path=tmp_path / "freeze.json")


def test_refuses_dirty_tree_on_freeze_and_on_open(tmp_path):
    with pytest.raises(seal.SealError, match="dirty"):
        seal.write_freeze(22, {}, clean=lambda: False, path=tmp_path / "f.json")
    p = tmp_path / "f.json"
    seal.write_freeze(22, {}, clean=lambda: True, path=p)
    with pytest.raises(seal.SealError, match="dirty"):
        seal.check_freeze(clean=lambda: False, path=p)
    assert seal.check_freeze(clean=lambda: True, path=p)["trial_count"] == 22


@pytest.mark.parametrize("field", ["config_sha256", "code_sha256"])
def test_refuses_hash_mismatch(tmp_path, field):
    p = tmp_path / "f.json"
    seal.write_freeze(22, {}, clean=lambda: True, path=p)
    rec = json.loads(p.read_text())
    rec[field] = "0" * 64
    p.write_text(json.dumps(rec))
    with pytest.raises(seal.SealError, match="changed since the freeze"):
        seal.check_freeze(clean=lambda: True, path=p)
