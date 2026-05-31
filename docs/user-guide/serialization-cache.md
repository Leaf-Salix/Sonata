# Serialization and Caching

## Overview

Sonata provides stable JSON serialization and fingerprint-based caching for
Scores, PlanHandles, and EligibilityResults. Serialization enables persistence,
debugging, and cross-process communication. Caching avoids repeating expensive
eligibility checks when the same static subgraph is encountered multiple times.

## Key Concepts

### Serialization Schema

All serialized payloads include a `schema_version` field for forward-compatible
deserialization. The current schema versions are:

| Payload | Schema Version Constant | Current Value |
|---------|------------------------|---------------|
| Score | `SCORE_SCHEMA_VERSION` | 1 |
| EligibilityResult | `ELIGIBILITY_RESULT_SCHEMA_VERSION` | 1 |
| PlanHandle | `PLAN_HANDLE_SCHEMA_VERSION` | 1 |
| Fingerprint | `FINGERPRINT_VERSION` | 1 |
| Cache | `CACHE_SCHEMA_VERSION` | 1 |

Attempting to deserialize a payload with a mismatched schema version raises
`DeserializationError`.

### Score Serialization

#### score_to_dict / score_to_json

Convert a `Score` to a JSON-serializable dictionary or a JSON string:

```python
from sonata import score_to_dict, score_to_json

# To dictionary (for programmatic use)
data = score_to_dict(score)
print(data["schema_version"])  # 1
print(data["name"])
print(len(data["tasks"]))

# To JSON string (for persistence or transport)
json_text = score_to_json(score)
print(json_text[:80])
```

The dictionary schema:

```json
{
  "schema_version": 1,
  "name": "my_kernel",
  "runtime_target": {
    "runtime": "host_build_graph",
    "function_name": "build_my_kernel_graph",
    "aicpu_thread_num": 4,
    "config_comment": []
  },
  "tasks": [
    {
      "task_id": 0,
      "func_id": 0,
      "core_type": "aic",
      "name": "matmul",
      "args": ["input_a", "input_b", "output_c"],
      "arg_directions": ["input", "input", "output"],
      "arg_storage_keys": ["param:input_a", "param:input_b", "alloc:output_c"]
    }
  ],
  "dependencies": [
    {"producer": 0, "consumer": 1, "kind": "data"}
  ],
  "shape_assumptions": [
    {"symbol": "input_tensor", "dims": [128, 256]}
  ],
  "metadata": {}
}
```

#### score_from_dict / score_from_json

Reconstruct a `Score` from a serialized form:

```python
from sonata import score_from_dict, score_from_json, score_to_dict, score_to_json

# Round-trip through dict
data = score_to_dict(score)
restored = score_from_dict(data)
assert restored.name == score.name
assert restored.task_count() == score.task_count()

# Round-trip through JSON
json_text = score_to_json(score)
restored = score_from_json(json_text)
assert restored.name == score.name
```

If the input is malformed or has a schema version mismatch,
`DeserializationError` is raised:

```python
from sonata import score_from_dict, DeserializationError

try:
    score = score_from_dict({"schema_version": 999, "name": "bad"})
except DeserializationError as e:
    print(f"Cannot deserialize: {e}")
```

### PlanHandle Serialization

PlanHandles serialize similarly, with additional fields for the function registry
and argument bindings:

```python
from sonata import plan_handle_to_dict, plan_handle_to_json
from sonata import plan_handle_from_dict, plan_handle_from_json

# Serialize
data = plan_handle_to_dict(plan_handle)
json_text = plan_handle_to_json(plan_handle)

# Deserialize
restored = plan_handle_from_dict(data)
restored = plan_handle_from_json(json_text)
```

### EligibilityResult Serialization

```python
from sonata import eligibility_result_to_dict
from sonata import eligibility_result_from_dict, eligibility_result_from_json

data = eligibility_result_to_dict(result)
restored = eligibility_result_from_dict(data)
```

### Score Fingerprint

The fingerprint is a stable SHA-256 hash that identifies a Score's computation
identity. Two Scores with the same tasks, dependencies, shape assumptions, and
name produce the same fingerprint, regardless of metadata differences.

```python
from sonata import score_fingerprint

fp = score_fingerprint(score)
print(f"Fingerprint: {fp[:16]}...")  # 64-character hex string

# Include metadata in the fingerprint (for audit/debug identity)
fp_with_meta = score_fingerprint(score, include_metadata=True)
```

**Fingerprint identity includes:**

- `name`
- `tasks` (all fields)
- `dependencies` (all fields)
- `shape_assumptions` (all fields)
- `metadata` (only when `include_metadata=True`)

**Fingerprint identity excludes:**

- `runtime_target` (this is an artifact identity field, not a computation identity field)

The fingerprint is used as the cache key and is embedded in `PlanHandle` to link
a runtime plan back to its originating Score.

### ScoreCache

`ScoreCache` is an in-memory, fingerprint-keyed cache for Scores and
PlanHandles.

```python
from sonata import ScoreCache, Score, score_fingerprint

cache = ScoreCache()

# Store a score
fp = cache.store(score)
print(f"Stored with fingerprint: {fp[:16]}...")

# Look up by fingerprint
payload = cache.lookup(fp)
if payload is not None:
    print("Cache hit!")
else:
    print("Cache miss.")

# Check membership
if cache.contains(fp):
    print("Score is cached.")

# Store a PlanHandle associated with the score
cache.store_plan_handle(plan_handle, fingerprint=fp)

# Look up the PlanHandle
ph_payload = cache.lookup_plan_handle(fp)
```

**Cache operations:**

| Method | Description |
|--------|-------------|
| `store(score)` | Store a Score, return its fingerprint. |
| `store_plan_handle(plan_handle)` | Associate a PlanHandle with a cached Score. |
| `lookup(fingerprint)` | Return Score payload dict or `None`. |
| `lookup_plan_handle(fingerprint)` | Return PlanHandle payload dict or `None`. |
| `contains(fingerprint)` | Return whether a valid entry exists. |
| `invalidate(*fingerprints)` | Remove entries by fingerprint. |
| `invalidate_all()` | Remove all entries. |
| `entry_count()` | Number of cached entries. |
| `stats()` | Cache hit/miss statistics. |

**Cache statistics:**

```python
stats = cache.stats()
print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
print(f"Hit rate: {stats['hit_rate_pct']}%")
print(f"Entries: {stats['entry_count']}")
```

**Cache persistence:**

```python
# Save to disk
cache.save("sonata_cache.json")

# Load from disk
cache = ScoreCache.load("sonata_cache.json")
```

**Schema-aware invalidation:** Lookups automatically return `None` (miss) when
the cached entry's `schema_version` does not match the current
`SCORE_SCHEMA_VERSION`. This provides forward-compatible invalidation without
manual cache clearing.

### cached_score

The `cached_score` function provides a look-up-or-build pattern:

```python
from sonata import ScoreCache, cached_score

cache = ScoreCache()

def build_my_score():
    # Expensive eligibility check and score construction
    from sonata import check_static_eligibility
    result = check_static_eligibility(ir_node)
    return result.score

score, fingerprint, was_cached = cached_score(
    cache,
    build_my_score,
    fingerprint_hint="abc123...",  # optional: skip builder if in cache
)

if was_cached:
    print(f"Loaded from cache: {fingerprint[:16]}...")
else:
    print(f"Built fresh: {fingerprint[:16]}...")
```

When `fingerprint_hint` is provided and present in the cache, the builder is not
called. Otherwise the builder runs, the result is stored, and `was_cached` is
`False`.

### CacheEntry

Each cache entry is a `CacheEntry` dataclass:

```python
from sonata import CacheEntry

# Fields available on a cache entry:
# entry.fingerprint         -- SHA-256 hex string
# entry.score_payload       -- dict (serialized Score)
# entry.schema_version      -- score schema version at storage time
# entry.fingerprint_version -- fingerprint algorithm version
# entry.created_at          -- Unix timestamp
# entry.plan_handle_payload -- optional dict (serialized PlanHandle)
# entry.metadata            -- user-defined metadata dict
```

## Usage

### Full Serialization Workflow

```python
from sonata import (
    check_static_eligibility,
    score_to_json,
    score_from_json,
    score_fingerprint,
    ScoreCache,
)

# 1. Check eligibility and obtain a Score
result = check_static_eligibility(ir_node)
if not result.eligible:
    raise RuntimeError("Not eligible for static planning")

score = result.score

# 2. Compute fingerprint for identity
fp = score_fingerprint(score)

# 3. Serialize for persistence
json_text = score_to_json(score)
with open("score.json", "w") as f:
    f.write(json_text)

# 4. Later: deserialize and verify identity
with open("score.json") as f:
    restored = score_from_json(f.read())

assert score_fingerprint(restored) == fp
```

### Caching in a Compilation Pipeline

```python
from sonata import ScoreCache, cached_score, check_static_eligibility

cache = ScoreCache()

def compile_graph(ir_node):
    def builder():
        result = check_static_eligibility(ir_node)
        if not result.eligible:
            raise ValueError("Ineligible")
        return result.score

    score, fp, was_cached = cached_score(cache, builder)
    if was_cached:
        print(f"Reusing cached score {fp[:16]}...")
    return score

# First call: builds from scratch
score1 = compile_graph(graph_a)

# Second call with same graph: cache hit
score2 = compile_graph(graph_a)
```

## Limitations and Known Constraints

- **In-memory only by default.** `ScoreCache` is an in-memory cache. Use
  `save()`/`load()` for disk persistence; there is no automatic eviction or TTL.
- **No concurrent access protection.** `ScoreCache` is not thread-safe. Use
  external synchronization if sharing across threads.
- **Fingerprint does not cover `runtime_target`.** Two Scores that differ only
  in `runtime_target` will share a fingerprint. This is intentional: the
  computation identity is the same, but the runtime artifact may differ.
- **Schema version mismatch is a hard error.** Deserializing a payload with a
  different schema version raises `DeserializationError`. There is no automatic
  migration.
- **Non-JSON-serializable values use `repr()`.** Arguments that are not standard
  JSON types (bool, int, float, str, list, dict, None) are converted via
  `repr()` during serialization. This is a best-effort fallback; round-tripping
  such values is not guaranteed.
