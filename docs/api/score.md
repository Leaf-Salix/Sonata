# Score Module

Source: `src/sonata/score.py`, `src/sonata/fallback.py`

The Score module defines the core data model for Sonata static execution plans. All types are frozen dataclasses and intentionally pure Python, independent from PyPTO IR objects.

---

## Classes

### `Score`

```python
@dataclass(frozen=True)
class Score:
    name: str
    runtime_target: RuntimeTarget
    tasks: tuple[Task, ...] = ()
    dependencies: tuple[Dependency, ...] = ()
    shape_assumptions: tuple[ShapeAssumption, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
```

Inspectible static execution plan emitted before target-specific codegen.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Human-readable identifier for the score. |
| `runtime_target` | `RuntimeTarget` | Runtime metadata (deprecated since v0.2 -- use `PlanHandle.runtime_target` instead). |
| `tasks` | `tuple[Task, ...]` | Ordered precomputed runtime tasks. |
| `dependencies` | `tuple[Dependency, ...]` | Explicit edges between tasks. |
| `shape_assumptions` | `tuple[ShapeAssumption, ...]` | Static shape facts defining the runtime validity domain. |
| `metadata` | `dict[str, Any]` | Audit and debug metadata. |

**Methods:**

| Method | Return Type | Description |
|--------|-------------|-------------|
| `task_count()` | `int` | Return the number of tasks in the score. |
| `dependency_count()` | `int` | Return the number of explicit dependency edges. |
| `validate()` | `EligibilityResult` | Validate basic score consistency for static planning. Checks name, task ID uniqueness, core types, dependency validity, cycle detection, and shape assumptions. |

---

### `Task`

```python
@dataclass(frozen=True)
class Task:
    task_id: int
    func_id: int
    core_type: str
    args: tuple[Any, ...] = ()
    arg_directions: tuple[str, ...] = ()
    arg_storage_keys: tuple[Any | None, ...] = ()
    name: str | None = None
```

One precomputed runtime task in a Sonata score.

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `int` | Unique non-negative task identifier. |
| `func_id` | `int` | Function identifier (maps to callee). |
| `core_type` | `str` | Target core type: `"aic"`, `"aiv"`, or `"mixed"`. |
| `args` | `tuple[Any, ...]` | Positional argument values or names. |
| `arg_directions` | `tuple[str, ...]` | Direction annotations per arg (e.g. `"input"`, `"output"`, `"inout"`). |
| `arg_storage_keys` | `tuple[Any \| None, ...]` | Storage identity keys per arg, or `None` when unknown. |
| `name` | `str \| None` | Optional callee function name. |

---

### `Dependency`

```python
@dataclass(frozen=True)
class Dependency:
    producer: int
    consumer: int
    kind: str = "data"
```

Explicit edge between two precomputed tasks.

| Field | Type | Description |
|-------|------|-------------|
| `producer` | `int` | Task ID of the producer. |
| `consumer` | `int` | Task ID of the consumer. |
| `kind` | `str` | Dependency semantics: `"data"` (RAW), `"storage"` (WAW), `"war"` (WAR), or `"ordering"` (pure ordering). |

---

### `ShapeAssumption`

```python
@dataclass(frozen=True)
class ShapeAssumption:
    symbol: str
    dims: tuple[int, ...]
```

Static shape fact that defines the runtime validity domain for a score.

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | `str` | Symbol name (e.g. a parameter name). Must be non-empty and unique within a score. |
| `dims` | `tuple[int, ...]` | Concrete positive integer dimensions. |

---

### `RuntimeTarget`

```python
@dataclass(frozen=True)
class RuntimeTarget:
    runtime: str = "tensormap_and_ringbuffer"
    function_name: str = "aicpu_orchestration_entry"
    aicpu_thread_num: int | None = 4
    config_comment: tuple[str, ...] = (
        "# Runtime configuration for tensormap_and_ringbuffer.",
        "# This runtime requires 4 AICPU threads (3 schedulers + 1 orchestrator on thread 3).",
    )
```

Runtime metadata emitted into generated `kernel_config.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `runtime` | `str` | `"tensormap_and_ringbuffer"` | Runtime identifier. |
| `function_name` | `str` | `"aicpu_orchestration_entry"` | Entry function name. |
| `aicpu_thread_num` | `int \| None` | `4` | Number of AICPU threads. |
| `config_comment` | `tuple[str, ...]` | *(see above)* | Comments emitted in config output. |

---

### `EligibilityResult`

```python
@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    score: Score | None = None
    reasons: tuple[str, ...] = ()
    reason_details: tuple[FallbackReason, ...] = ()
```

Result of checking whether a score or IR region can use Sonata.

| Field | Type | Description |
|-------|------|-------------|
| `eligible` | `bool` | Whether the input is eligible for Sonata planning. |
| `score` | `Score \| None` | The constructed Score when eligible. |
| `reasons` | `tuple[str, ...]` | Human-readable rejection reason messages. |
| `reason_details` | `tuple[FallbackReason, ...]` | Structured reason entries with codes and severity. |

**Class Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `accept` | `accept(score: Score) -> EligibilityResult` | Build an eligible result for `score`. |
| `reject` | `reject(*reasons: str \| FallbackReason) -> EligibilityResult` | Build an ineligible result with one or more reasons. |
| `accept_with_warnings` | `accept_with_warnings(score: Score, *warnings: str \| FallbackReason) -> EligibilityResult` | Build an eligible result with warning-level detail entries. |

**Instance Methods:**

| Method | Return Type | Description |
|--------|-------------|-------------|
| `has_errors()` | `bool` | Return whether any reason detail has severity `"error"`. |
| `has_warnings()` | `bool` | Return whether any reason detail has severity `"warning"`. |

---

### `FallbackReason`

```python
@dataclass(frozen=True)
class FallbackReason:
    code: str
    message: str
    severity: str = "error"
```

Structured explanation for why a score or region is ineligible.

| Field | Type | Description |
|-------|------|-------------|
| `code` | `str` | Stable `FallbackCode` value or best-effort slug. Safe as a routing key. |
| `message` | `str` | Human-readable explanation. |
| `severity` | `str` | `"error"` or `"warning"`. |

---

### `FallbackCode`

Source: `src/sonata/fallback.py`

```python
class FallbackCode(str, Enum):
```

Stable reason codes for eligibility rejection and validation failure. These codes are safe to use as routing or filtering keys.

| Constant | Value |
|----------|-------|
| `UNSUPPORTED_ROOT_KIND` | `"unsupported_root_kind"` |
| `CONTROL_FLOW_NOT_SUPPORTED` | `"control_flow_not_supported"` |
| `UNSUPPORTED_RUNTIME_SCOPE` | `"unsupported_runtime_scope"` |
| `TENSOR_READ_NOT_SUPPORTED` | `"tensor_read_not_supported"` |
| `ENTRY_FUNCTION_NOT_ORCHESTRATION` | `"entry_function_not_orchestration"` |
| `UNSUPPORTED_PYPTO_ADAPTER_SCOPE` | `"unsupported_pypto_adapter_scope"` |
| `SCORE_VALIDATION_FAILED` | `"score_validation_failed"` |
| `STORAGE_COVERAGE_BELOW_THRESHOLD` | `"storage_coverage_below_threshold"` |
| `UNKNOWN_MEMORY_STORAGE_CRITICAL` | `"unknown_memory_storage_critical"` |
| `DATAFLOW_DIRECTIONS_UNAVAILABLE` | `"dataflow_directions_unavailable"` |
| `DATAFLOW_DIRECTIONS_INCOMPLETE` | `"dataflow_directions_incomplete"` |
| `RUNTIME_ADAPTER_FINGERPRINT_MISMATCH` | `"runtime_adapter_fingerprint_mismatch"` |
| `RUNTIME_ADAPTER_CONTRACT_VERSION_MISMATCH` | `"runtime_adapter_contract_version_mismatch"` |
| `RUNTIME_ADAPTER_FUNC_NOT_REGISTERED` | `"runtime_adapter_func_not_registered"` |
| `RUNTIME_ADAPTER_FUNC_UNREFERENCED` | `"runtime_adapter_func_unreferenced"` |
| `RUNTIME_ADAPTER_BINDING_INCOMPLETE` | `"runtime_adapter_binding_incomplete"` |
| `RUNTIME_ADAPTER_INVALID_EDGE` | `"runtime_adapter_invalid_edge"` |

---

## Constants

### `DEFAULT_RUNTIME_TARGET`

```python
DEFAULT_RUNTIME_TARGET = RuntimeTarget()
```

Default `RuntimeTarget` instance with `tensormap_and_ringbuffer` runtime and 4 AICPU threads.

---

## Functions

### `is_static_shape_dim`

```python
def is_static_shape_dim(dim: Any) -> bool
```

Return whether `dim` is a positive concrete shape dimension (a positive `int`, not `bool`).
