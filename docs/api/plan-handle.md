# PlanHandle / Runtime Module

Source: `src/sonata/plan_handle.py`, `src/sonata/runtime_adapter.py`

PlanHandle carries the runtime artifact key information that was previously embedded in Score. Score remains the computation identity; PlanHandle bridges Score to a specific runtime target, adapter stage, and function registry.

---

## Schema Version Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `PLAN_HANDLE_SCHEMA_VERSION` | `1` | Schema version for PlanHandle serialization. |
| `RUNTIME_CONTRACT_VERSION` | `1` | Runtime contract version for adapter compatibility checks. |

---

## Classes

### `PlanHandle`

```python
@dataclass(frozen=True)
class PlanHandle:
    score_fingerprint: str
    runtime_target: RuntimeTarget
    source_adapter: str
    runtime_contract_version: int = RUNTIME_CONTRACT_VERSION
    func_registry: FuncRegistry = field(default_factory=FuncRegistry)
    arg_bindings: tuple[RuntimeArgBinding, ...] = ()
    schema_version: int = PLAN_HANDLE_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
```

Runtime artifact key for a Sonata static plan. Bridges a Score (computation identity) to a specific runtime target, adapter stage, and function registry.

| Field | Type | Description |
|-------|------|-------------|
| `score_fingerprint` | `str` | SHA-256 fingerprint of the associated Score. |
| `runtime_target` | `RuntimeTarget` | Runtime target metadata. |
| `source_adapter` | `str` | Name of the adapter stage that produced this plan (e.g. `"post_simplify"`). |
| `runtime_contract_version` | `int` | Contract version for adapter compatibility validation. |
| `func_registry` | `FuncRegistry` | Function name to runtime ID mapping. |
| `arg_bindings` | `tuple[RuntimeArgBinding, ...]` | Task argument bindings from Sonata storage identity to runtime handles. |
| `schema_version` | `int` | PlanHandle schema version. |
| `metadata` | `dict[str, Any]` | Additional metadata. |

**Class Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `from_score` | `from_score(score: Score, *, source_adapter: str = "post_simplify", runtime_target: RuntimeTarget \| None = None) -> PlanHandle` | Build a minimal PlanHandle from a Score. Uses `score_fingerprint()` and `FuncRegistry.from_score()`. |

---

### `FuncRegistry`

```python
@dataclass(frozen=True)
class FuncRegistry:
    entries: tuple[FuncRegistryEntry, ...] = ()
```

Explicit function name to runtime ID mapping. Uses function name as the logical key rather than extraction order.

| Field | Type | Description |
|-------|------|-------------|
| `entries` | `tuple[FuncRegistryEntry, ...]` | Registry entries. |

**Methods:**

| Method | Return Type | Description |
|--------|-------------|-------------|
| `by_name(name: str)` | `FuncRegistryEntry \| None` | Look up an entry by function name. |
| `names()` | `frozenset[str]` | Return all registered function names. |
| `sonata_func_ids()` | `dict[str, int]` | Return name-to-Sonata-func_id mapping. |
| `runtime_func_ids()` | `dict[str, int \| None]` | Return name-to-runtime-func_id mapping. |
| `has_unbound_entries()` | `bool` | `True` if any entry has `runtime_func_id is None`. |
| `bind_runtime_ids(func_name_to_id: dict[str, int])` | `FuncRegistry` | Return a new registry with runtime IDs bound from a name-to-id map. |
| `from_score(score: Score)` *(classmethod)* | `FuncRegistry` | Build a registry from Score tasks, using callee names as keys. |

---

### `FuncRegistryEntry`

```python
@dataclass(frozen=True)
class FuncRegistryEntry:
    name: str
    sonata_func_id: int
    runtime_func_id: int | None = None
```

One entry in the function registry.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Function name (logical key). |
| `sonata_func_id` | `int` | Sonata-internal function identifier. |
| `runtime_func_id` | `int \| None` | Runtime-assigned function ID, or `None` if unbound. |

---

### `RuntimeArgBinding`

```python
@dataclass(frozen=True)
class RuntimeArgBinding:
    task_id: int
    arg_index: int
    storage_key: str | None
    direction: str
    runtime_handle: Any
```

Maps a task arg from Sonata storage identity to a runtime handle.

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `int` | Task ID owning the argument. |
| `arg_index` | `int` | Positional index within the task's args. |
| `storage_key` | `str \| None` | Sonata storage key for the argument. |
| `direction` | `str` | Argument direction (e.g. `"input"`, `"output"`). |
| `runtime_handle` | `Any` | Opaque runtime handle (integer index, tensor handle, etc.). |

---

## Runtime Adapter Types

Source: `src/sonata/runtime_adapter.py`

### `HostBuildGraphRuntimeAdapter`

```python
class HostBuildGraphRuntimeAdapter:
```

Generate and validate host-build-graph runtime plans. Translates Score + PlanHandle into a `HostBuildGraphPlan`, performing structural validation.

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `generate` | `generate(score: Score, plan_handle: PlanHandle) -> RuntimeAdapterResult` | Generate a `HostBuildGraphPlan` from Score and PlanHandle. Validates first; returns rejection reasons on failure. |
| `validate` | `validate(score: Score, plan_handle: PlanHandle) -> tuple[FallbackReason, ...]` | Validate Score/PlanHandle structural consistency. Checks fingerprint match, contract version, func registry completeness, arg binding completeness, and dependency edge validity. |

---

### `HostBuildGraphPlan`

```python
@dataclass(frozen=True)
class HostBuildGraphPlan:
    tasks: tuple[HostBuildGraphTask, ...] = ()
    edges: tuple[HostBuildGraphEdge, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
```

Structured output for the `host_build_graph` runtime.

| Field | Type | Description |
|-------|------|-------------|
| `tasks` | `tuple[HostBuildGraphTask, ...]` | Task table for the runtime graph builder. |
| `edges` | `tuple[HostBuildGraphEdge, ...]` | Explicit dependency edges. |
| `metadata` | `dict[str, Any]` | Audit metadata (fingerprint, source adapter, contract version, etc.). |

**Methods:**

| Method | Return Type | Description |
|--------|-------------|-------------|
| `task_count()` | `int` | Number of tasks. |
| `edge_count()` | `int` | Number of edges. |

---

### `HostBuildGraphTask`

```python
@dataclass(frozen=True)
class HostBuildGraphTask:
    task_id: int
    func_id: int
    core_type: str
    runtime_args: tuple[Any, ...] = ()
    name: str | None = None
```

One task in a host-build-graph runtime plan. `runtime_args` are resolved from `RuntimeArgBinding` handles.

---

### `HostBuildGraphEdge`

```python
@dataclass(frozen=True)
class HostBuildGraphEdge:
    producer: int
    consumer: int
```

One explicit dependency edge in a host-build-graph runtime plan.

---

### `RuntimeAdapterResult`

```python
@dataclass(frozen=True)
class RuntimeAdapterResult:
    success: bool
    plan: HostBuildGraphPlan | None = None
    reasons: tuple[FallbackReason, ...] = ()
```

Result of a runtime adapter generation or validation.

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether generation succeeded. |
| `plan` | `HostBuildGraphPlan \| None` | The generated plan when successful. |
| `reasons` | `tuple[FallbackReason, ...]` | Structured failure reasons. |

**Class Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `accept` | `accept(plan: HostBuildGraphPlan) -> RuntimeAdapterResult` | Build a successful result. |
| `reject` | `reject(*reasons: FallbackReason) -> RuntimeAdapterResult` | Build a failed result with reasons. |
