# Storage / Alias / Liveness Module

Source: `src/sonata/storage.py`, `src/sonata/alias.py`, `src/sonata/liveness.py`, `src/sonata/memory_plan.py`

Storage-key extraction, alias analysis, buffer liveness computation, and graph-level memory planning for Sonata's storage model.

---

## Storage Coverage Thresholds

Source: `src/sonata/storage.py`

| Constant | Value | Description |
|----------|-------|-------------|
| `STORAGE_COVERAGE_WARN_THRESHOLD` | `0.5` | Below this ratio of known/total storage keys, eligibility emits a warning. |
| `STORAGE_COVERAGE_REJECT_THRESHOLD` | `0.0` | Reserved rejection threshold (not yet enforced as a hard reject). |

---

## Storage Functions

### `collect_storage_keys`

```python
def collect_storage_keys(
    node: Any,
    *,
    walk: Callable[[Any], Iterable[Any]],
    kind: Callable[[Any], str],
    call_name: Callable[[Any], str | None],
    is_builtin_call: Callable[[str], bool],
    arg_name: Callable[[Any], str],
    arg_directions: Callable[[Any], tuple[str, ...]],
) -> dict[int, str]
```

Return structural storage keys keyed by stable Var identity. Walks assignment statements and propagates storage keys through `tensor.create` calls, tuple get-item expressions, and call output assignments.

### `collect_call_output_vars`

```python
def collect_call_output_vars(
    node: Any,
    *,
    walk: Callable[[Any], Iterable[Any]],
    kind: Callable[[Any], str],
) -> dict[int, Any]
```

Return assignment LHS variables keyed by assigned Call object identity.

### `arg_storage_keys`

```python
def arg_storage_keys(call: Any, storage_keys: dict[int, str]) -> tuple[str | None, ...]
```

Return storage keys for each positional call argument when known.

### `propagate_call_output_storage`

```python
def propagate_call_output_storage(
    output_var: Any,
    call: Any,
    storage_keys: dict[int, str],
    *,
    arg_directions: Callable[[Any], tuple[str, ...]],
) -> None
```

Propagate the first known write-arg storage key to an assigned call output. Mutates `storage_keys` in place.

### `call_write_storage_keys`

```python
def call_write_storage_keys(
    call: Any,
    storage_keys: dict[int, str],
    *,
    arg_directions: Callable[[Any], tuple[str, ...]],
) -> tuple[str | None, ...]
```

Return storage keys for write-like call args in positional order.

### `storage_key`

```python
def storage_key(arg: Any, storage_keys: dict[int, str]) -> str | None
```

Return the known storage key for `arg` if one has been collected.

---

## Alias Analysis

Source: `src/sonata/alias.py`

### Alias Relation Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `ALIAS_DISJOINT` | `"disjoint"` | Keys refer to independent buffers. |
| `ALIAS_ALIAS` | `"alias"` | Keys refer to the same underlying buffer. |
| `ALIAS_VIEW` | `"view"` | One key is a sub-view of the other. |
| `ALIAS_INPLACE` | `"inplace"` | Keys share storage in an in-place operation. |

### `AliasRelation`

```python
@dataclass(frozen=True)
class AliasRelation:
    key_a: str
    key_b: str
    relation: str
```

Alias relationship between two storage keys.

| Field | Type | Description |
|-------|------|-------------|
| `key_a` | `str` | First storage key. |
| `key_b` | `str` | Second storage key. |
| `relation` | `str` | One of the `ALIAS_*` constants. |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `is_disjoint` | `bool` | `True` if `relation == ALIAS_DISJOINT`. |
| `shares_storage` | `bool` | `True` if relation is `alias`, `view`, or `inplace`. |

### `analyze_aliases`

```python
def analyze_aliases(
    storage_keys: tuple[str, ...],
    alias_declarations: dict[str, str] | None = None,
    view_declarations: dict[str, str] | None = None,
    inplace_declarations: set[tuple[str, str]] | None = None,
) -> tuple[AliasRelation, ...]
```

Analyze alias relationships between storage keys.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `storage_keys` | `tuple[str, ...]` | *(required)* | Storage keys to analyze. |
| `alias_declarations` | `dict[str, str] \| None` | `None` | Maps a key to its canonical alias target. |
| `view_declarations` | `dict[str, str] \| None` | `None` | Maps a view key to its source key. |
| `inplace_declarations` | `set[tuple[str, str]] \| None` | `None` | `(key_a, key_b)` pairs that share storage in-place. |

Keys with no declared relationship are assumed disjoint. Returns one `AliasRelation` per unique pair (sorted order).

---

## Buffer Liveness

Source: `src/sonata/liveness.py`

### `BufferLifetime`

```python
@dataclass(frozen=True)
class BufferLifetime:
    storage_key: str
    birth: int
    death: int
```

Lifetime interval for one storage buffer across the task sequence.

| Field | Type | Description |
|-------|------|-------------|
| `storage_key` | `str` | The storage key identifying the buffer. |
| `birth` | `int` | First task_id that writes the buffer. |
| `death` | `int` | Last task_id that reads or writes the buffer. |

**Methods:**

| Method | Return Type | Description |
|--------|-------------|-------------|
| `overlaps(other)` | `bool` | Return whether two lifetimes overlap (both alive at the same time). |

### `StorageConflict`

```python
@dataclass(frozen=True)
class StorageConflict:
    key_a: str
    key_b: str
```

Two buffers that are simultaneously alive and cannot share memory.

### `compute_lifetimes`

```python
def compute_lifetimes(tasks: tuple[Task, ...]) -> tuple[BufferLifetime, ...]
```

Compute buffer lifetimes from task args, directions, and storage keys.

- **Birth** = first `task_id` that writes the buffer.
- **Death** = last `task_id` that reads or writes the buffer.
- Only memory-direction args with known storage keys contribute.

### `find_conflicts`

```python
def find_conflicts(
    lifetimes: tuple[BufferLifetime, ...],
) -> tuple[StorageConflict, ...]
```

Find pairs of buffers with overlapping lifetimes. Returns sorted conflict pairs.

---

## Memory Planning

Source: `src/sonata/memory_plan.py`

### `BufferAllocation`

```python
@dataclass(frozen=True)
class BufferAllocation:
    storage_key: str
    offset: int
    size: int
```

One buffer's memory allocation.

| Field | Type | Description |
|-------|------|-------------|
| `storage_key` | `str` | The storage key identifying the buffer. |
| `offset` | `int` | Byte offset in the memory plan. |
| `size` | `int` | Buffer size in bytes. |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `end` | `int` | `offset + size` (exclusive end). |

### `MemoryPlan`

```python
@dataclass(frozen=True)
class MemoryPlan:
    allocations: tuple[BufferAllocation, ...] = ()
    peak_memory: int = 0
```

Memory layout plan for a Score's buffers.

| Field | Type | Description |
|-------|------|-------------|
| `allocations` | `tuple[BufferAllocation, ...]` | Per-buffer allocations, sorted by storage key. |
| `peak_memory` | `int` | Maximum memory offset used (high-water mark). |

**Methods:**

| Method | Return Type | Description |
|--------|-------------|-------------|
| `total_allocated()` | `int` | Sum of all allocation sizes. |
| `by_key(storage_key)` | `BufferAllocation \| None` | Look up allocation by storage key. |

### `plan_memory`

```python
def plan_memory(
    lifetimes: tuple[BufferLifetime, ...],
    buffer_sizes: dict[str, int],
) -> MemoryPlan
```

Plan memory layout using greedy-first-fit by size.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lifetimes` | `tuple[BufferLifetime, ...]` | Buffer lifetimes from `compute_lifetimes`. |
| `buffer_sizes` | `dict[str, int]` | Maps storage keys to their byte sizes. |

Buffers are sorted by size descending and placed at the first available offset that does not conflict with overlapping lifetimes. Buffers with non-overlapping lifetimes can share memory offsets.
