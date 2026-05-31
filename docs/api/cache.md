# Cache Module

Source: `src/sonata/cache.py`

Fingerprint-based Score and PlanHandle cache. Avoids repeating expensive eligibility checks and Score construction when the same static subgraph is encountered multiple times. Entries are keyed by `score_fingerprint()` and carry schema version metadata for forward-compatible invalidation.

---

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `CACHE_SCHEMA_VERSION` | `1` | Schema version for cache serialization. |

---

## Classes

### `CacheEntry`

```python
@dataclass(frozen=True)
class CacheEntry:
    fingerprint: str
    score_payload: dict[str, Any]
    schema_version: int = SCORE_SCHEMA_VERSION
    fingerprint_version: int = FINGERPRINT_VERSION
    created_at: float = field(default_factory=time)
    plan_handle_payload: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

One cached Score with optional PlanHandle association.

| Field | Type | Description |
|-------|------|-------------|
| `fingerprint` | `str` | SHA-256 fingerprint of the score. |
| `score_payload` | `dict[str, Any]` | Serialized score dictionary. |
| `schema_version` | `int` | Score schema version at time of caching. |
| `fingerprint_version` | `int` | Fingerprint algorithm version. |
| `created_at` | `float` | Unix timestamp of entry creation. |
| `plan_handle_payload` | `dict[str, Any] \| None` | Serialized plan handle, if associated. |
| `metadata` | `dict[str, Any]` | Additional cache entry metadata. |

---

### `ScoreCache`

```python
class ScoreCache:
```

In-memory fingerprint-keyed cache for Scores and PlanHandles.

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `store` | `store(score: Score, *, fingerprint: str \| None = None) -> str` | Store a Score and return its fingerprint. Preserves existing PlanHandle association if re-storing. |
| `store_plan_handle` | `store_plan_handle(plan_handle: PlanHandle, *, fingerprint: str \| None = None) -> str` | Store a PlanHandle associated with a Score fingerprint. Raises `KeyError` if no cached score exists for the fingerprint. |
| `lookup` | `lookup(fingerprint: str) -> dict[str, Any] \| None` | Return the cached Score payload or `None` on miss. Increments hit/miss counters. Returns `None` if schema version does not match. |
| `lookup_plan_handle` | `lookup_plan_handle(fingerprint: str) -> dict[str, Any] \| None` | Return the cached PlanHandle payload or `None` on miss. |
| `contains` | `contains(fingerprint: str) -> bool` | Return whether a valid entry exists for `fingerprint` (matching schema version). Does not affect hit/miss counters. |
| `invalidate` | `invalidate(*fingerprints: str) -> int` | Remove entries by fingerprint. Returns count of removed entries. |
| `invalidate_all` | `invalidate_all() -> int` | Remove all entries. Returns count of removed entries. |
| `entry_count` | `entry_count() -> int` | Return the number of cached entries. |
| `stats` | `stats() -> dict[str, int]` | Return cache statistics: `entry_count`, `hits`, `misses`, `total_lookups`, `hit_rate_pct`. |
| `to_dict` | `to_dict() -> dict[str, Any]` | Serialize the cache to a JSON-compatible dictionary. |
| `from_dict` | `from_dict(data: dict[str, Any]) -> ScoreCache` *(classmethod)* | Restore a cache from a dictionary produced by `to_dict`. |
| `save` | `save(path: str \| Path) -> Path` | Persist the cache to a JSON file. Creates parent directories. Returns the resolved path. |
| `load` | `load(path: str \| Path) -> ScoreCache` *(classmethod)* | Load a cache from a JSON file. |

---

## Functions

### `cached_score`

```python
def cached_score(
    cache: ScoreCache,
    score_builder: Callable[[], Score],
    *,
    fingerprint_hint: str | None = None,
) -> tuple[Score, str, bool]
```

Look up or build a Score, returning `(score, fingerprint, was_cached)`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache` | `ScoreCache` | *(required)* | The cache to look up in and store to. |
| `score_builder` | `Callable[[], Score]` | *(required)* | A zero-argument callable that builds the Score. |
| `fingerprint_hint` | `str \| None` | `None` | If provided and present in the cache, the cached payload is returned without calling `score_builder`. |

**Returns:** `tuple[Score, str, bool]` -- the score, its fingerprint, and whether it was served from cache.

If `fingerprint_hint` is provided and found in the cache, the cached payload is deserialized and returned (`was_cached=True`). Otherwise the builder is called, the result is stored, and `was_cached=False`.
