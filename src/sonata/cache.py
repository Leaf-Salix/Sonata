# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fingerprint-based Score and PlanHandle cache.

The cache avoids repeating expensive eligibility checks and Score construction
when the same static subgraph is encountered multiple times.  Entries are keyed
by :func:`~sonata.serialization.score_fingerprint` and carry schema version
metadata for forward-compatible invalidation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Callable, TYPE_CHECKING

from .deserialization import score_from_dict as _score_from_dict
from .plan_handle import GuardStatus
from .score import Score
from .serialization import (
    FINGERPRINT_VERSION,
    SCORE_SCHEMA_VERSION,
    score_fingerprint,
    score_to_dict,
    score_to_json,
)

if TYPE_CHECKING:
    from .plan_handle import PlanHandle

CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CacheEntry:
    """One cached Score with optional PlanHandle association and guard status."""

    fingerprint: str
    score_payload: dict[str, Any]
    schema_version: int = SCORE_SCHEMA_VERSION
    fingerprint_version: int = FINGERPRINT_VERSION
    created_at: float = field(default_factory=time)
    plan_handle_payload: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    guard_status: GuardStatus = GuardStatus.ALL_SATISFIED


class ScoreCache:
    """In-memory fingerprint-keyed cache for Scores and PlanHandles."""

    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}
        self._hits: int = 0
        self._misses: int = 0

    def store(self, score: Score, *, fingerprint: str | None = None, guard_status: GuardStatus = GuardStatus.ALL_SATISFIED) -> str:
        """Store a Score and return its fingerprint.
        
        Args:
            score: The Score to cache
            fingerprint: Optional fingerprint hint
            guard_status: Initial guard status (default: ALL_SATISFIED)
        """
        fp = fingerprint or score_fingerprint(score)
        payload = score_to_dict(score)
        existing = self._entries.get(fp)
        plan_payload = existing.plan_handle_payload if existing else None
        meta = existing.metadata if existing else {}
        gs = existing.guard_status if existing else guard_status
        self._entries[fp] = CacheEntry(
            fingerprint=fp,
            score_payload=payload,
            schema_version=SCORE_SCHEMA_VERSION,
            fingerprint_version=FINGERPRINT_VERSION,
            plan_handle_payload=plan_payload,
            metadata=meta,
            guard_status=gs,
        )
        return fp

    def store_plan_handle(self, plan_handle: "PlanHandle", *, fingerprint: str | None = None) -> str:
        """Store a PlanHandle associated with a Score fingerprint."""
        from .plan_handle import PlanHandle as PH
        from .serialization import plan_handle_to_dict

        fp = fingerprint or plan_handle.score_fingerprint
        existing = self._entries.get(fp)
        if existing is None:
            raise KeyError(f"no cached score for fingerprint: {fp[:16]}...")
        self._entries[fp] = CacheEntry(
            fingerprint=existing.fingerprint,
            score_payload=existing.score_payload,
            schema_version=existing.schema_version,
            fingerprint_version=existing.fingerprint_version,
            created_at=existing.created_at,
            plan_handle_payload=plan_handle_to_dict(plan_handle),
            metadata=existing.metadata,
            guard_status=plan_handle.guard_status,
        )
        return fp

    def lookup(self, fingerprint: str) -> dict[str, Any] | None:
        """Return the cached Score payload or None on miss.
        
        Treats guard violations (PARTIAL_FAILED or ALL_FAILED) as cache misses.
        """
        entry = self._entries.get(fingerprint)
        if entry is None:
            self._misses += 1
            return None
        if entry.schema_version != SCORE_SCHEMA_VERSION:
            self._misses += 1
            return None
        # Phase 5 E2: Guard validation - treat violations as cache miss
        if entry.guard_status != GuardStatus.ALL_SATISFIED:
            self._misses += 1
            return None
        self._hits += 1
        return entry.score_payload

    def lookup_plan_handle(self, fingerprint: str) -> dict[str, Any] | None:
        """Return the cached PlanHandle payload or None on miss.
        
        Treats guard violations as cache misses.
        """
        entry = self._entries.get(fingerprint)
        if entry is None or entry.plan_handle_payload is None:
            self._misses += 1
            return None
        # Phase 5 E2: Guard validation for plan handle lookup
        if entry.guard_status != GuardStatus.ALL_SATISFIED:
            self._misses += 1
            return None
        self._hits += 1
        return entry.plan_handle_payload

    def contains(self, fingerprint: str) -> bool:
        """Return whether a valid entry exists for ``fingerprint``.
        
        Returns False if guard status is not ALL_SATISFIED.
        """
        entry = self._entries.get(fingerprint)
        if entry is None:
            return False
        if entry.schema_version != SCORE_SCHEMA_VERSION:
            return False
        # Phase 5 E2: Guard validation for contains check
        if entry.guard_status != GuardStatus.ALL_SATISFIED:
            return False
        return True

    def invalidate(self, *fingerprints: str) -> int:
        """Remove entries by fingerprint. Returns count of removed entries."""
        removed = 0
        for fp in fingerprints:
            if fp in self._entries:
                del self._entries[fp]
                removed += 1
        return removed

    def invalidate_all(self) -> int:
        """Remove all entries. Returns count of removed entries."""
        count = len(self._entries)
        self._entries.clear()
        return count

    def entry_count(self) -> int:
        """Return the number of cached entries."""
        return len(self._entries)

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        total = self._hits + self._misses
        return {
            "entry_count": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "total_lookups": total,
            "hit_rate_pct": round(self._hits * 100 / total) if total > 0 else 0,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the cache to a JSON-compatible dictionary."""
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "entries": {
                fp: _entry_to_dict(entry) for fp, entry in sorted(self._entries.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoreCache":
        """Restore a cache from a dictionary produced by :meth:`to_dict`."""
        cache = cls()
        entries = data.get("entries", {})
        for fp, entry_data in entries.items():
            cache._entries[fp] = _entry_from_dict(entry_data)
        return cache

    def save(self, path: str | Path) -> Path:
        """Persist the cache to a JSON file. Returns the resolved path."""
        import json

        resolved = Path(path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return resolved

    @classmethod
    def load(cls, path: str | Path) -> "ScoreCache":
        """Load a cache from a JSON file."""
        import json

        resolved = Path(path).resolve()
        data = json.loads(resolved.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def cached_score(
    cache: ScoreCache,
    score_builder: Callable[[], Score],
    *,
    fingerprint_hint: str | None = None,
) -> tuple[Score, str, bool]:
    """Look up or build a Score, returning (score, fingerprint, was_cached).

    If ``fingerprint_hint`` is provided and present in the cache, the cached
    payload is returned without calling ``score_builder``.  Otherwise the
    builder is called, the result is stored, and ``was_cached`` is False.
    """
    if fingerprint_hint is not None:
        payload = cache.lookup(fingerprint_hint)
        if payload is not None:
            score = _score_from_dict(payload)
            return score, fingerprint_hint, True

    score = score_builder()
    fp = cache.store(score)
    return score, fp, False


def _entry_to_dict(entry: CacheEntry) -> dict[str, Any]:
    return {
        "fingerprint": entry.fingerprint,
        "score_payload": entry.score_payload,
        "schema_version": entry.schema_version,
        "fingerprint_version": entry.fingerprint_version,
        "created_at": entry.created_at,
        "plan_handle_payload": entry.plan_handle_payload,
        "metadata": entry.metadata,
        "guard_status": entry.guard_status.value,
    }


def _entry_from_dict(data: dict[str, Any]) -> CacheEntry:
    # Phase 5 E1: Deserialize guard_status with backward compatibility
    guard_status_value = data.get("guard_status", "all_satisfied")
    try:
        guard_status = GuardStatus(guard_status_value)
    except ValueError:
        guard_status = GuardStatus.ALL_SATISFIED  # Default for unknown values
    
    return CacheEntry(
        fingerprint=data["fingerprint"],
        score_payload=data["score_payload"],
        schema_version=data.get("schema_version", SCORE_SCHEMA_VERSION),
        fingerprint_version=data.get("fingerprint_version", FINGERPRINT_VERSION),
        created_at=data.get("created_at", 0.0),
        plan_handle_payload=data.get("plan_handle_payload"),
        metadata=data.get("metadata", {}),
        guard_status=guard_status,
    )


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CacheEntry",
    "ScoreCache",
    "cached_score",
]
