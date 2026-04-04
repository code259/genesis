from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from core.provider_runtime import runtime_dir


def cache_lookup(namespace: str, payload: dict, version_parts: list[str] | None = None) -> dict | None:
    path = cache_path(namespace, payload, version_parts=version_parts)
    if not path.exists():
        _record_cache_stat("miss")
        return None
    _record_cache_stat("hit")
    return json.loads(path.read_text())


def cache_store(namespace: str, payload: dict, value: dict, version_parts: list[str] | None = None) -> Path:
    path = cache_path(namespace, payload, version_parts=version_parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))
    return path


def cache_path(namespace: str, payload: dict, version_parts: list[str] | None = None) -> Path:
    version = "|".join(version_parts or [])
    digest = sha256((version + "\n" + _normalize_payload(payload)).encode("utf-8")).hexdigest()
    return runtime_dir() / "cache" / namespace / f"{digest}.json"


def cache_summary() -> dict:
    stats_path = runtime_dir() / "cache" / "stats.json"
    if not stats_path.exists():
        return {"hits": 0, "misses": 0}
    return json.loads(stats_path.read_text())


def _record_cache_stat(name: str):
    stats_path = runtime_dir() / "cache" / "stats.json"
    stats = {"hits": 0, "misses": 0}
    if stats_path.exists():
        stats.update(json.loads(stats_path.read_text()))
    field = "hits" if name == "hit" else "misses"
    stats[field] = stats.get(field, 0) + 1
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))


def _normalize_payload(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
