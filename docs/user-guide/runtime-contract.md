# Runtime Contract

## Overview

The runtime contract defines the pipeline from a validated Score to a concrete
runtime execution plan. The central carrier is the **PlanHandle**, which bridges
a Score (computation identity) to a specific runtime target, adapter stage, and
function registry. The `HostBuildGraphRuntimeAdapter` then translates a
Score + PlanHandle pair into a `HostBuildGraphPlan` -- the task table and edge
table consumed by the host-side graph builder.

The pipeline is:

```
Score --> PlanHandle --> HostBuildGraphRuntimeAdapter --> HostBuildGraphPlan
```

## Key Concepts

### PlanHandle

A `PlanHandle` is the runtime artifact key for a Sonata static plan. It carries
all the runtime-specific metadata that was previously embedded in Score, keeping
Score focused on computation identity.

```python
from sonata import PlanHandle, Score, RuntimeTarget, score_fingerprint

# Build from a Score (recommended)
plan_handle = PlanHandle.from_score(
    score,
    source_adapter="post_simplify",
    runtime_target=RuntimeTarget(
        runtime="host_build_graph",
        function_name="build_my_kernel_graph",
    ),
)

print(f"Score fingerprint: {plan_handle.score_fingerprint[:16]}...")
print(f"Source adapter:    {plan_handle.source_adapter}")
print(f"Contract version: {plan_handle.runtime_contract_version}")
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `score_fingerprint` | `str` | SHA-256 fingerprint linking back to the originating Score. |
| `runtime_target` | `RuntimeTarget` | Runtime metadata (replaces deprecated `Score.runtime_target`). |
| `source_adapter` | `str` | Name of the PyPTO adapter stage that produced the Score. |
| `runtime_contract_version` | `int` | Contract version (currently `RUNTIME_CONTRACT_VERSION = 1`). |
| `func_registry` | `FuncRegistry` | Function name to ID mapping. |
| `arg_bindings` | `tuple[RuntimeArgBinding, ...]` | Task argument to runtime handle mappings. |
| `schema_version` | `int` | PlanHandle schema version. |
| `metadata` | `dict[str, Any]` | Extensible metadata. |

### FuncRegistry

The `FuncRegistry` maps callee function names to Sonata internal IDs and
optional runtime IDs. It uses function name as the logical key rather than
extraction order.

```python
from sonata import FuncRegistry, FuncRegistryEntry, Score

# Build from a Score's tasks
registry = FuncRegistry.from_score(score)

# Look up by name
entry = registry.by_name("matmul")
if entry is not None:
    print(f"Sonata func_id: {entry.sonata_func_id}")
    print(f"Runtime func_id: {entry.runtime_func_id}")

# Query registered names
print(registry.names())  # frozenset({"matmul", "activation", ...})

# Check for unbound entries
if registry.has_unbound_entries():
    print("Some functions have no runtime ID yet.")
```

**FuncRegistryEntry fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Callee function name (logical key). |
| `sonata_func_id` | `int` | Internal Sonata function ID. |
| `runtime_func_id` | `int | None` | Runtime-assigned function ID, or `None` if unbound. |

**Binding runtime IDs:**

When the PyPTO codegen layer assigns function IDs, bind them to the registry:

```python
# Map from PyPTO codegen: function name -> runtime ID
func_name_to_id = {
    "matmul": 0,
    "activation": 1,
    "feedforward": 2,
}

bound_registry = registry.bind_runtime_ids(func_name_to_id)

# All entries should now have runtime_func_id
assert not bound_registry.has_unbound_entries()
```

### RuntimeArgBinding

A `RuntimeArgBinding` maps a task argument from Sonata's storage identity to a
concrete runtime handle.

```python
from sonata import RuntimeArgBinding

binding = RuntimeArgBinding(
    task_id=0,
    arg_index=2,
    storage_key="alloc:output_c",
    direction="output",
    runtime_handle=42,  # opaque runtime tensor handle
)
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `int` | The task this binding applies to. |
| `arg_index` | `int` | Positional argument index within the task. |
| `storage_key` | `str | None` | Sonata storage identity for the argument. |
| `direction` | `str` | Argument direction (`"input"`, `"output"`, etc.). |
| `runtime_handle` | `Any` | Opaque runtime handle (integer index, tensor handle, etc.). |

### HostBuildGraphRuntimeAdapter

The `HostBuildGraphRuntimeAdapter` translates a Score + PlanHandle into a
`HostBuildGraphPlan` suitable for the host-side graph builder runtime.

```python
from sonata import HostBuildGraphRuntimeAdapter

adapter = HostBuildGraphRuntimeAdapter()

# Generate a runtime plan
result = adapter.generate(score, plan_handle)

if result.success:
    plan = result.plan
    print(f"Runtime tasks: {plan.task_count()}")
    print(f"Runtime edges: {plan.edge_count()}")
    for task in plan.tasks:
        print(f"  Task {task.task_id}: func_id={task.func_id}, core={task.core_type}")
    for edge in plan.edges:
        print(f"  Edge: {edge.producer} -> {edge.consumer}")
else:
    for reason in result.reasons:
        print(f"Adapter error: [{reason.code}] {reason.message}")
```

### RuntimeAdapterResult

The result of adapter generation or validation:

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether the plan was generated successfully. |
| `plan` | `HostBuildGraphPlan | None` | The generated plan when successful. |
| `reasons` | `tuple[FallbackReason, ...]` | Structured error reasons when failed. |

### HostBuildGraphPlan

The concrete runtime plan:

```python
from sonata import HostBuildGraphPlan, HostBuildGraphTask, HostBuildGraphEdge

# Inspect the plan
plan = result.plan

for task in plan.tasks:
    print(f"Task {task.task_id}:")
    print(f"  func_id:     {task.func_id}")
    print(f"  core_type:   {task.core_type}")
    print(f"  runtime_args: {task.runtime_args}")

for edge in plan.edges:
    print(f"Edge: {edge.producer} -> {edge.consumer}")
```

**HostBuildGraphTask fields:**

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `int` | Task identifier (matches Score task_id). |
| `func_id` | `int` | Runtime function ID (from FuncRegistry if bound). |
| `core_type` | `str` | Target core type. |
| `runtime_args` | `tuple[Any, ...]` | Resolved runtime handles for each argument. |
| `name` | `str | None` | Callee function name. |

**HostBuildGraphEdge fields:**

| Field | Type | Description |
|-------|------|-------------|
| `producer` | `int` | Producer task_id. |
| `consumer` | `int` | Consumer task_id. |

## Usage

### Full Pipeline: Score to Runtime Plan

```python
from sonata import (
    check_static_eligibility,
    PlanHandle,
    FuncRegistry,
    RuntimeArgBinding,
    RuntimeTarget,
    HostBuildGraphRuntimeAdapter,
)

# 1. Obtain a Score from eligibility checking
result = check_static_eligibility(ir_node)
assert result.eligible
score = result.score

# 2. Build a PlanHandle
plan_handle = PlanHandle.from_score(
    score,
    source_adapter="post_simplify",
    runtime_target=RuntimeTarget(
        runtime="host_build_graph",
        function_name="build_my_graph",
    ),
)

# 3. Bind runtime function IDs (from PyPTO codegen)
func_name_to_id = {"matmul": 0, "activation": 1}
bound_registry = plan_handle.func_registry.bind_runtime_ids(func_name_to_id)

# 4. Create argument bindings
arg_bindings = (
    RuntimeArgBinding(task_id=0, arg_index=0, storage_key="param:input",
                      direction="input", runtime_handle=100),
    RuntimeArgBinding(task_id=0, arg_index=1, storage_key="param:weight",
                      direction="input", runtime_handle=101),
    RuntimeArgBinding(task_id=0, arg_index=2, storage_key="alloc:output",
                      direction="output", runtime_handle=102),
    # ... one binding per memory-direction arg per task
)

# 5. Rebuild PlanHandle with bound registry and bindings
from dataclasses import replace
plan_handle = replace(
    plan_handle,
    func_registry=bound_registry,
    arg_bindings=arg_bindings,
)

# 6. Generate the runtime plan
adapter = HostBuildGraphRuntimeAdapter()
adapter_result = adapter.generate(score, plan_handle)

if adapter_result.success:
    plan = adapter_result.plan
    # Pass plan.tasks and plan.edges to the host-side graph builder
else:
    for reason in adapter_result.reasons:
        print(f"Failed: {reason.message}")
```

### Validation Without Generation

You can validate structural consistency without generating a plan:

```python
adapter = HostBuildGraphRuntimeAdapter()
reasons = adapter.validate(score, plan_handle)

if reasons:
    for r in reasons:
        print(f"[{r.code}] {r.message}")
else:
    print("Score and PlanHandle are structurally consistent.")
```

**Validation checks performed:**

1. **Fingerprint match.** PlanHandle's `score_fingerprint` matches the Score's
   computed fingerprint.
2. **Contract version.** PlanHandle's `runtime_contract_version` matches the
   adapter's expected version.
3. **Function registration.** Every named Score task has a corresponding
   FuncRegistry entry, and every registry entry is referenced by at least one
   task.
4. **Argument binding completeness.** Every memory-direction argument has a
   corresponding `RuntimeArgBinding`.
5. **Edge validity.** All dependency producer/consumer IDs reference existing
   tasks.

## Limitations and Known Constraints

- **PlanHandle is frozen.** Use `dataclasses.replace()` to create modified
  copies rather than mutating fields.
- **`runtime_handle` is opaque.** The PlanHandle layer does not interpret
  runtime handles. Their semantics are defined by the downstream runtime.
- **No automatic binding generation.** `RuntimeArgBinding` instances must be
  constructed by the integration layer that has access to runtime tensor
  handles. Sonata does not generate them automatically.
- **Contract version is strict.** A PlanHandle with a different
  `runtime_contract_version` than the adapter will be rejected. There is no
  automatic version negotiation.
- **Single runtime target.** Each `HostBuildGraphRuntimeAdapter` generates plans
  for the `host_build_graph` runtime only. Other runtime targets would require
  separate adapter implementations.
