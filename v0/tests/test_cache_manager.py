import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core import cache_manager


def test_cache_lookup_records_hit_and_miss(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("GENESIS_RUNTIME_DIR", str(tmp_path / "runtime"))
    payload = {"task": "S1T1", "value": 1}

    assert cache_manager.cache_lookup("demo", payload) is None
    cache_manager.cache_store("demo", payload, {"value": {"ok": True}})
    cached = cache_manager.cache_lookup("demo", payload)

    assert cached == {"value": {"ok": True}}
    summary = cache_manager.cache_summary()
    assert summary["hits"] >= 1
    assert summary["misses"] >= 1
