# Regions and Multi-Adapter

## Overview

Sonata v0.4 introduces region-level eligibility, replacing the v0.1 whole-graph
all-or-nothing check. An IR graph is split into **static** and **dynamic**
regions: static regions can be independently planned by Sonata, while dynamic
regions fall back to the original PyPTO runtime path. The multi-adapter
registry provides metadata about available PyPTO pipeline stages and their
capabilities, enabling flexible adapter selection.

## Key Concepts

### Region

A `Region` is a contiguous segment of an IR graph, classified as either static
or dynamic.

```python
from sonata import Region, REGION_STATIC, REGION_DYNAMIC

# Regions are produced by extract_regions(), not constructed directly
# Inspect a region:
print(f"Region {region.region_id}: {region.kind}")
print(f"  Nodes: {region.node_count}")
print(f"  Is static: {region.is_static}")
print(f"  Is dynamic: {region.is_dynamic}")

if region.is_dynamic and region.fallback_reason:
    print(f"  Fallback: [{region.fallback_reason.code}] {region.fallback_reason.message}")
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `region_id` | `int` | Unique identifier within the region map. |
| `kind` | `str` | `"static"` or `"dynamic"`. |
| `nodes` | `tuple[Any, ...]` | IR nodes in this region. |
| `control_flow_kind` | `str | None` | For dynamic regions, the kind of control flow that caused the split (e.g., `"ForStmt"`, `"IfStmt"`). |
| `fallback_reason` | `FallbackReason | None` | Structured reason for dynamic classification. |

### RegionMap

A `RegionMap` is the collection of all regions extracted from an IR graph.

```python
from sonata import extract_regions, RegionMap

region_map = extract_regions(ir_node)

# Query region statistics
print(f"Total regions:   {len(region_map.regions)}")
print(f"Static regions:  {len(region_map.static_regions())}")
print(f"Dynamic regions: {len(region_map.dynamic_regions())}")
print(f"Static ratio:    {region_map.static_ratio():.1%}")
print(f"All static:      {region_map.all_static()}")
print(f"All dynamic:     {region_map.all_dynamic()}")
```

**Methods:**

| Method | Description |
|--------|-------------|
| `static_regions()` | Return only static regions. |
| `dynamic_regions()` | Return only dynamic regions. |
| `static_ratio()` | Fraction of regions that are static (0.0 to 1.0). |
| `all_static()` | `True` if every region is static. |
| `all_dynamic()` | `True` if every region is dynamic. |

### extract_regions

The `extract_regions` function walks an IR node's body and splits statements
into contiguous runs of static statements separated by dynamic statements:

```python
from sonata import extract_regions

region_map = extract_regions(function_node)

for region in region_map.regions:
    if region.is_static:
        print(f"Region {region.region_id}: STATIC ({region.node_count} nodes)")
    else:
        print(f"Region {region.region_id}: DYNAMIC "
              f"(control_flow={region.control_flow_kind})")
```

**Splitting rules:**

1. Statements whose class name is `ForStmt`, `IfStmt`, or `WhileStmt` become
   individual dynamic regions with `control_flow_kind` set accordingly.
2. `RuntimeScopeStmt` statements also become individual dynamic regions.
3. All other statements are accumulated into contiguous static regions.
4. Static regions are flushed whenever a dynamic statement is encountered.

Example: an IR body with statements `[assign, assign, for_loop, assign, if_stmt, assign]`
would produce:

```
Region 0: static   (2 nodes: assign, assign)
Region 1: dynamic  (1 node: for_loop, control_flow_kind="ForStmt")
Region 2: static   (1 node: assign)
Region 3: dynamic  (1 node: if_stmt, control_flow_kind="IfStmt")
Region 4: static   (1 node: assign)
```

### check_region_eligibility

Region-level eligibility checking combines region extraction with static
eligibility:

```python
from sonata import check_region_eligibility

result = check_region_eligibility(ir_node, entry_name="main")

if result.eligible:
    score = result.score
    print(f"Partially eligible: {score.task_count()} tasks")
else:
    for detail in result.reason_details:
        print(f"[{detail.code}] {detail.message}")
```

**Behavior:**

- If all regions are dynamic, the result is rejected with the dynamic regions'
  fallback reasons.
- If at least one static region exists, the full
  `check_static_eligibility()` is invoked on the node (the static portion of
  the graph).
- If the graph has a mix of static and dynamic regions, the static portions are
  planned and the dynamic portions fall back to the PyPTO runtime.

### AdapterDescriptor

An `AdapterDescriptor` describes one PyPTO pipeline adapter stage:

```python
from sonata import AdapterDescriptor, AdapterCapability

adapter = AdapterDescriptor(
    name="post_simplify",
    version=1,
    capabilities=AdapterCapability(
        static_shapes=True,
        storage_keys=True,
        arg_directions=True,
        dependency_kinds=False,
        control_flow_regions=False,
        runtime_scopes=False,
    ),
    description="Post-Simplify adapter after CollectCommGroups.",
    certified_dump="after_collect_comm_groups_simplify",
)

# Serialize
data = adapter.to_dict()

# Deserialize
restored = AdapterDescriptor.from_dict(data)
```

**AdapterDescriptor fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique adapter name. |
| `version` | `int` | Adapter version number. |
| `capabilities` | `AdapterCapability` | What the adapter can provide. |
| `description` | `str` | Human-readable description. |
| `certified_dump` | `str` | Certified dump stage name. |

### AdapterCapability

Declares what an adapter stage can provide:

```python
from sonata import AdapterCapability

caps = AdapterCapability(
    static_shapes=True,          # Can extract static tensor shapes
    storage_keys=True,           # Can extract storage key identities
    arg_directions=True,         # Can extract argument directions
    dependency_kinds=False,      # Cannot classify dependency kinds
    control_flow_regions=False,  # Cannot extract control flow regions
    runtime_scopes=False,        # Cannot extract runtime scope facts
)
```

| Capability | Default | Description |
|------------|---------|-------------|
| `static_shapes` | `True` | Adapter extracts static tensor shapes. |
| `storage_keys` | `True` | Adapter extracts storage key identities. |
| `arg_directions` | `True` | Adapter extracts argument directions. |
| `dependency_kinds` | `False` | Adapter can classify RAW/WAW/WAR dependencies. |
| `control_flow_regions` | `False` | Adapter extracts control flow regions. |
| `runtime_scopes` | `False` | Adapter extracts RuntimeScopeStmt facts. |

### AdapterRegistry

The `AdapterRegistry` manages available adapter stages:

```python
from sonata import AdapterRegistry, AdapterDescriptor, default_registry

# Use the pre-populated default registry
registry = default_registry()

# Or build a custom registry
registry = AdapterRegistry()
registry.register(AdapterDescriptor(
    name="my_adapter",
    version=1,
    description="Custom adapter",
))
```

**Built-in adapters** (in `default_registry()`):

| Name | Capabilities | Certified Dump |
|------|-------------|----------------|
| `post_simplify` | static_shapes, storage_keys, arg_directions | `after_collect_comm_groups_simplify` |
| `pre_runtime` | static_shapes, storage_keys, arg_directions, dependency_kinds | `pre_runtime_scope` |
| `post_simplify_with_scope` | static_shapes, storage_keys, arg_directions, runtime_scopes | `after_collect_comm_groups_simplify_with_scope` |

**Registry operations:**

```python
# Look up by name
adapter = registry.get("post_simplify")
if adapter is not None:
    print(f"Found: {adapter.description}")

# List all registered names
print(registry.names())  # frozenset({"post_simplify", "pre_runtime", ...})

# Count
print(f"Adapters: {registry.entry_count()}")

# Select by required capabilities
dataflow_capable = registry.select(dependency_kinds=True)
for a in dataflow_capable:
    print(f"{a.name}: {a.description}")

# Validate a PlanHandle's source_adapter reference
from sonata import PlanHandle
is_valid = registry.validate_plan_handle(plan_handle)
```

**Capability-based selection:**

The `select()` method filters adapters by required capability flags. Only
`True` values act as filters:

```python
# Find adapters that support runtime scopes
scope_adapters = registry.select(runtime_scopes=True)

# Find adapters that support both dependency kinds and static shapes
advanced = registry.select(dependency_kinds=True, static_shapes=True)
```

**Registry serialization:**

```python
# Serialize
data = registry.to_dict()

# Deserialize
restored = AdapterRegistry.from_dict(data)
```

## Usage

### Region-Aware Compilation Pipeline

```python
from sonata import extract_regions, check_region_eligibility

# 1. Extract regions
region_map = extract_regions(ir_node)

# 2. Check if any static regions exist
if region_map.all_dynamic():
    print("Entire graph is dynamic; using PyPTO runtime.")
else:
    # 3. Run region-level eligibility
    result = check_region_eligibility(ir_node, entry_name="main")

    if result.eligible:
        score = result.score
        print(f"Static plan: {score.task_count()} tasks")

        # Dynamic regions are handled by PyPTO runtime
        for region in region_map.dynamic_regions():
            print(f"Dynamic region {region.region_id}: {region.control_flow_kind}")
    else:
        print("Region eligibility failed:")
        for detail in result.reason_details:
            print(f"  [{detail.code}] {detail.message}")
```

### Selecting the Right Adapter

```python
from sonata import default_registry

registry = default_registry()

# If you need dataflow dependency analysis:
dataflow_adapters = registry.select(dependency_kinds=True)
if dataflow_adapters:
    adapter = dataflow_adapters[0]
    print(f"Using adapter: {adapter.name} (certified dump: {adapter.certified_dump})")

# If you need runtime scope awareness:
scope_adapters = registry.select(runtime_scopes=True)

# Verify that a PlanHandle references a known adapter
if not registry.validate_plan_handle(plan_handle):
    print(f"Unknown source adapter: {plan_handle.source_adapter}")
```

### Custom Adapter Registration

```python
from sonata import AdapterRegistry, AdapterDescriptor, AdapterCapability

registry = AdapterRegistry()

# Register the standard adapters
from sonata import POST_SIMPLIFY, PRE_RUNTIME, POST_SIMPLIFY_WITH_SCOPE
registry.register(POST_SIMPLIFY)
registry.register(PRE_RUNTIME)
registry.register(POST_SIMPLIFY_WITH_SCOPE)

# Register a custom adapter
registry.register(AdapterDescriptor(
    name="post_materialize",
    version=1,
    capabilities=AdapterCapability(
        static_shapes=True,
        storage_keys=True,
        arg_directions=True,
        dependency_kinds=True,
        runtime_scopes=True,
    ),
    description="Post-MaterializeRuntimeScopes adapter with full capabilities.",
    certified_dump="after_materialize_runtime_scopes",
))

# Select the most capable adapter
full = registry.select(
    dependency_kinds=True,
    runtime_scopes=True,
)
print(f"Fully capable adapters: {[a.name for a in full]}")
```

## Limitations and Known Constraints

- **Region splitting is syntactic.** Regions are split by statement class name,
  not by semantic analysis. A sequence of apparently static statements that
  depends on a dynamic loop variable is still classified as static.
- **No nested region extraction.** Control flow nodes are treated as opaque
  dynamic regions. Their internal structure is not further decomposed into
  sub-regions.
- **Region eligibility delegates to whole-graph check.** When at least one
  static region exists, `check_region_eligibility` delegates to
  `check_static_eligibility` on the whole node. It does not yet plan individual
  regions independently.
- **Adapter registry is in-memory only.** There is no automatic discovery of
  adapters; all must be explicitly registered.
- **Capability flags are boolean.** There is no partial capability or
  quality-level distinction. An adapter either provides a capability or it does
  not.
- **`register()` rejects duplicates.** Registering an adapter with an already
  registered name raises `ValueError`. Unregister or use a different name to
  replace an adapter.
