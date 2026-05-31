# Adapters Module

Source: `src/sonata/adapters.py`

Multi-adapter registry for PyPTO pipeline stages. Each adapter represents a specific PyPTO pipeline stage that can produce normalized facts for Sonata Score construction. The registry tracks available adapters, their capabilities, and validates PlanHandle adapter references.

---

## Pre-defined Adapter Descriptors

### `POST_SIMPLIFY`

```python
POST_SIMPLIFY = AdapterDescriptor(
    name="post_simplify",
    version=1,
    capabilities=AdapterCapability(
        static_shapes=True,
        storage_keys=True,
        arg_directions=True,
    ),
    description="Post-Simplify adapter after CollectCommGroups, before MaterializeRuntimeScopes.",
    certified_dump="after_collect_comm_groups_simplify",
)
```

The primary v0.1 adapter. Operates on the certified dump after CollectCommGroups and the final Simplify pass.

### `PRE_RUNTIME`

```python
PRE_RUNTIME = AdapterDescriptor(
    name="pre_runtime",
    version=1,
    capabilities=AdapterCapability(
        static_shapes=True,
        storage_keys=True,
        arg_directions=True,
        dependency_kinds=True,
    ),
    description="Pre-runtime adapter with dependency kind awareness.",
    certified_dump="pre_runtime_scope",
)
```

Pre-runtime adapter with dependency kind awareness.

### `POST_SIMPLIFY_WITH_SCOPE`

```python
POST_SIMPLIFY_WITH_SCOPE = AdapterDescriptor(
    name="post_simplify_with_scope",
    version=1,
    capabilities=AdapterCapability(
        static_shapes=True,
        storage_keys=True,
        arg_directions=True,
        runtime_scopes=True,
    ),
    description="Post-Simplify adapter that also extracts RuntimeScopeStmt facts.",
    certified_dump="after_collect_comm_groups_simplify_with_scope",
)
```

Post-Simplify adapter that also extracts `RuntimeScopeStmt` facts.

---

## Classes

### `AdapterCapability`

```python
@dataclass(frozen=True)
class AdapterCapability:
    static_shapes: bool = True
    storage_keys: bool = True
    arg_directions: bool = True
    dependency_kinds: bool = False
    control_flow_regions: bool = False
    runtime_scopes: bool = False
```

Declares what an adapter stage can provide.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `static_shapes` | `bool` | `True` | Can extract static shape assumptions. |
| `storage_keys` | `bool` | `True` | Can extract storage keys for memory analysis. |
| `arg_directions` | `bool` | `True` | Can extract argument direction annotations. |
| `dependency_kinds` | `bool` | `False` | Can classify dependency edges by hazard kind. |
| `control_flow_regions` | `bool` | `False` | Can extract control flow regions. |
| `runtime_scopes` | `bool` | `False` | Can extract `RuntimeScopeStmt` facts. |

---

### `AdapterDescriptor`

```python
@dataclass(frozen=True)
class AdapterDescriptor:
    name: str
    version: int = 1
    capabilities: AdapterCapability = field(default_factory=AdapterCapability)
    description: str = ""
    certified_dump: str = ""
```

Metadata for one PyPTO pipeline adapter stage.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique adapter name (used as `PlanHandle.source_adapter`). |
| `version` | `int` | Adapter version number. |
| `capabilities` | `AdapterCapability` | What this adapter can provide. |
| `description` | `str` | Human-readable description. |
| `certified_dump` | `str` | Name of the certified PyPTO dump this adapter expects. |

**Methods:**

| Method | Return Type | Description |
|--------|-------------|-------------|
| `to_dict()` | `dict[str, Any]` | Serialize to a JSON-compatible dictionary. |
| `from_dict(data)` *(classmethod)* | `AdapterDescriptor` | Deserialize from a dictionary. |

---

### `AdapterRegistry`

```python
class AdapterRegistry:
```

Registry of available PyPTO adapter stages.

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `register` | `register(adapter: AdapterDescriptor) -> None` | Register an adapter. Raises `ValueError` on duplicate name. |
| `get` | `get(name: str) -> AdapterDescriptor \| None` | Look up an adapter by name. Returns `None` if not found. |
| `names` | `names() -> frozenset[str]` | Return all registered adapter names. |
| `adapters` | `adapters() -> tuple[AdapterDescriptor, ...]` | Return all registered adapters. |
| `entry_count` | `entry_count() -> int` | Return the number of registered adapters. |
| `select` | `select(**required_capabilities: bool) -> tuple[AdapterDescriptor, ...]` | Return adapters that satisfy all required capability flags. Only `True` values act as filters. |
| `validate_plan_handle` | `validate_plan_handle(plan_handle: PlanHandle) -> bool` | Return whether the PlanHandle's `source_adapter` is registered. |
| `to_dict` | `to_dict() -> dict[str, Any]` | Serialize the registry to a dictionary. |
| `from_dict` | `from_dict(data: dict[str, Any]) -> AdapterRegistry` *(classmethod)* | Restore a registry from a dictionary. |

---

## Functions

### `default_registry`

```python
def default_registry() -> AdapterRegistry
```

Return a registry pre-populated with the standard Sonata adapters: `POST_SIMPLIFY`, `PRE_RUNTIME`, and `POST_SIMPLIFY_WITH_SCOPE`.

### `select_adapter`

There is no standalone `select_adapter` function. Use `AdapterRegistry.select()` instead:

```python
registry = default_registry()
matches = registry.select(runtime_scopes=True)
```
