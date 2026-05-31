# Eligibility Module

Source: `src/sonata/eligibility.py`, `src/sonata/regions.py`

Conservative static-eligibility checks that determine whether an IR graph or region can use Sonata planning. Uses structural introspection instead of importing IR classes.

---

## Functions

### `check_static_eligibility`

```python
def check_static_eligibility(
    node: Any,
    *,
    runtime_target: RuntimeTarget | None = None,
    entry_name: str | None = None,
    dependency_policy: str = DEPENDENCY_POLICY_SEQUENTIAL_V0,
    require_certified: bool = False,
) -> EligibilityResult
```

Return whether `node` is eligible for an initial Sonata static score.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `node` | `Any` | *(required)* | A PyPTO IR node (Function or Program). |
| `runtime_target` | `RuntimeTarget \| None` | `None` | Runtime target metadata. When `None`, a `host_build_graph` target is generated from the node name. |
| `entry_name` | `str \| None` | `None` | If set, only the function with this name is considered as the orchestration entry point. |
| `dependency_policy` | `str` | `"sequential_v0"` | Dependency construction policy. Use `DEPENDENCY_POLICY_SEQUENTIAL_V0` or `DEPENDENCY_POLICY_DATAFLOW_V0`. |
| `require_certified` | `bool` | `False` | When `True`, enforce certified dump contract (stricter direction checks). |

**Returns:** `EligibilityResult`

**Behavior:**

1. Validates the root node kind (must be `Function` or `Program`).
2. Walks extraction roots and rejects control flow nodes (`ForStmt`, `IfStmt`, `WhileStmt`), runtime scope statements, and unsupported adapter kinds.
3. Rejects `tensor.read` calls.
4. Normalizes IR facts via `PostSimplifyPyPTOInputAdapter`.
5. Constructs tasks, dependencies, and shape assumptions.
6. Resolves the dependency policy -- falls back from `dataflow_v0` to `sequential_v0` when directions are incomplete.
7. Validates the constructed Score and checks storage coverage.

---

### `check_region_eligibility`

```python
def check_region_eligibility(
    node: Any,
    *,
    entry_name: str | None = None,
) -> EligibilityResult
```

Check eligibility at region granularity (v0.4).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `node` | `Any` | *(required)* | A PyPTO IR node. |
| `entry_name` | `str \| None` | `None` | Entry function name filter. |

**Returns:** `EligibilityResult`

**Behavior:**

- Extracts static and dynamic regions from the node.
- If all regions are dynamic, rejects with structured warnings.
- If at least one static region exists, delegates to `check_static_eligibility`.

---

### `extract_regions`

```python
def extract_regions(node: Any) -> RegionMap
```

Extract static and dynamic regions from an IR node.

| Parameter | Type | Description |
|-----------|------|-------------|
| `node` | `Any` | A PyPTO IR node with a body (list of statements). |

**Returns:** `RegionMap`

Walks the node's body and splits statements into contiguous runs of static statements (no control flow) separated by dynamic statements (control flow nodes like `ForStmt`, `IfStmt`, `WhileStmt`, or `RuntimeScopeStmt`).

---

## Region Types

### `Region`

```python
@dataclass(frozen=True)
class Region:
    region_id: int
    kind: str
    nodes: tuple[Any, ...] = ()
    control_flow_kind: str | None = None
    fallback_reason: FallbackReason | None = None
```

One region of an IR graph, classified as static or dynamic.

| Field | Type | Description |
|-------|------|-------------|
| `region_id` | `int` | Unique region identifier. |
| `kind` | `str` | `"static"` or `"dynamic"`. |
| `nodes` | `tuple[Any, ...]` | IR nodes belonging to this region. |
| `control_flow_kind` | `str \| None` | For dynamic regions, the kind of control flow (e.g. `"ForStmt"`). |
| `fallback_reason` | `FallbackReason \| None` | Structured reason for dynamic fallback. |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `is_static` | `bool` | `True` if `kind == "static"`. |
| `is_dynamic` | `bool` | `True` if `kind == "dynamic"`. |
| `node_count` | `int` | Number of nodes in the region. |

---

### `RegionMap`

```python
@dataclass(frozen=True)
class RegionMap:
    regions: tuple[Region, ...] = ()
```

Collection of regions extracted from an IR graph.

**Methods:**

| Method | Return Type | Description |
|--------|-------------|-------------|
| `static_regions()` | `tuple[Region, ...]` | Return only static regions. |
| `dynamic_regions()` | `tuple[Region, ...]` | Return only dynamic regions. |
| `static_ratio()` | `float` | Fraction of regions that are static (0.0 if no regions). |
| `all_static()` | `bool` | `True` if every region is static. |
| `all_dynamic()` | `bool` | `True` if every region is dynamic. |

---

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `REGION_STATIC` | `"static"` | Region kind for static (plannable) regions. |
| `REGION_DYNAMIC` | `"dynamic"` | Region kind for dynamic (fallback) regions. |
