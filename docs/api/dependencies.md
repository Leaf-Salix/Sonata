# Dependencies Module

Source: `src/sonata/dependencies.py`

Dependency policy builders for Sonata scores. Operates only on the pure-Python Task model, keeping dependency experiments decoupled from PyPTO's C++ IR bindings.

---

## Policy Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEPENDENCY_POLICY_SEQUENTIAL_V0` | `"sequential_v0"` | Chain tasks in extraction order (default). |
| `DEPENDENCY_POLICY_DATAFLOW_V0` | `"dataflow_v0"` | Build conservative RAW/WAW/WAR edges from task args and directions. |

---

## Functions

### `build_dependencies`

```python
def build_dependencies(
    tasks: tuple[Task, ...],
    *,
    policy: str = DEPENDENCY_POLICY_SEQUENTIAL_V0,
) -> tuple[Dependency, ...]
```

Build task dependencies according to a named Sonata policy.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tasks` | `tuple[Task, ...]` | *(required)* | The tasks to build dependencies for. |
| `policy` | `str` | `"sequential_v0"` | Policy name. Must be `"sequential_v0"` or `"dataflow_v0"`. |

**Returns:** `tuple[Dependency, ...]`

**Raises:** `ValueError` if `policy` is not a recognized policy name.

---

### `build_sequential_dependencies`

```python
def build_sequential_dependencies(tasks: tuple[Task, ...]) -> tuple[Dependency, ...]
```

Chain tasks in extraction order. Each task depends on the previous one with `kind="data"`.

---

### `build_dataflow_dependencies`

```python
def build_dataflow_dependencies(tasks: tuple[Task, ...]) -> tuple[Dependency, ...]
```

Build conservative RAW/WAW/WAR edges from task args and directions.

Every task must carry `arg_directions`. If directions are unavailable, callers should keep using `sequential_v0`.

**Edge classification:**

| Kind | Hazard | Description |
|------|--------|-------------|
| `"data"` | RAW (read-after-write) | Reader depends on prior writer. |
| `"storage"` | WAW (write-after-write) | Writer depends on prior writer. |
| `"war"` | WAR (write-after-read) | Writer depends on prior reader. |

When multiple edge kinds exist between the same producer/consumer pair, the more specific kind wins (priority: `data` > `storage` > `war` > `ordering`).

Storage keys are used as access keys when available; otherwise the arg value itself is used.

**Raises:** `ValueError` if tasks lack complete `arg_directions`.

---

### `build_mixed_dependencies`

```python
def build_mixed_dependencies(
    tasks: tuple[Task, ...],
    *,
    side_effect_tasks: frozenset[int] | None = None,
) -> tuple[Dependency, ...]
```

Build dataflow dependencies plus ordering edges for side-effect tasks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tasks` | `tuple[Task, ...]` | *(required)* | The tasks to build dependencies for. |
| `side_effect_tasks` | `frozenset[int] \| None` | `None` | If set, only these task IDs get ordering edges. Otherwise all tasks are considered. |

Dataflow edges carry their natural kind (`data`/`storage`/`war`). Ordering edges (`kind="ordering"`) are added between side-effect tasks that have no dataflow edge between them.

---

### `build_ordering_dependencies`

```python
def build_ordering_dependencies(
    tasks: tuple[Task, ...],
    *,
    side_effect_tasks: frozenset[int] | None = None,
) -> tuple[Dependency, ...]
```

Build pure ordering constraints between tasks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tasks` | `tuple[Task, ...]` | *(required)* | The tasks to chain. |
| `side_effect_tasks` | `frozenset[int] \| None` | `None` | If set, only tasks in this set get ordering edges. Otherwise all tasks are chained. |

All edges have `kind="ordering"`.

---

### `supports_dataflow_dependencies`

```python
def supports_dataflow_dependencies(tasks: tuple[Task, ...]) -> bool
```

Return whether all tasks carry enough direction data for `dataflow_v0`. Returns `True` when every task has `arg_directions` matching its `args` length.

---

### `dataflow_dependency_fallback_code`

```python
def dataflow_dependency_fallback_code(tasks: tuple[Task, ...]) -> FallbackCode | None
```

Return `None` when all tasks carry enough direction data, or a `FallbackCode` explaining why dataflow dependencies cannot be built.

| Return Value | Condition |
|--------------|-----------|
| `None` | All tasks have complete `arg_directions`. |
| `FallbackCode.DATAFLOW_DIRECTIONS_UNAVAILABLE` | No task has any `arg_directions`. |
| `FallbackCode.DATAFLOW_DIRECTIONS_INCOMPLETE` | Some but not all tasks have complete directions. |
