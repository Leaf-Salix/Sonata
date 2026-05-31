# Score Concepts

## Overview

A **Score** is the central data structure in Sonata. It represents a pre-computed
static execution plan -- a task dependency calendar that is generated at compile
time rather than computed dynamically at runtime. The Score captures the complete
scheduling picture: what tasks to execute, in what order, with what data
dependencies, and under what shape assumptions.

Sonata's Score model is intentionally pure Python and independent from PyPTO IR
objects. This keeps the scheduling layer decoupled from C++ bindings and makes
Scores easy to construct, inspect, serialize, and test in isolation.

## Key Concepts

### Score

The `Score` dataclass is the top-level container for a static execution plan.

```python
from sonata import Score, RuntimeTarget, Task, Dependency, ShapeAssumption

score = Score(
    name="my_kernel",
    runtime_target=RuntimeTarget(
        runtime="host_build_graph",
        function_name="build_my_kernel_graph",
    ),
    tasks=(
        Task(task_id=0, func_id=0, core_type="aic", name="matmul"),
        Task(task_id=1, func_id=1, core_type="aiv", name="activation"),
    ),
    dependencies=(
        Dependency(producer=0, consumer=1, kind="data"),
    ),
    shape_assumptions=(
        ShapeAssumption(symbol="input_tensor", dims=(128, 256)),
    ),
)
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Human-readable identifier for the score. Must not be empty. |
| `runtime_target` | `RuntimeTarget` | Runtime metadata (deprecated on Score since v0.2; prefer `PlanHandle.runtime_target`). |
| `tasks` | `tuple[Task, ...]` | Ordered collection of pre-computed runtime tasks. |
| `dependencies` | `tuple[Dependency, ...]` | Explicit edges between tasks. |
| `shape_assumptions` | `tuple[ShapeAssumption, ...]` | Static shape facts that define validity domain. |
| `metadata` | `dict[str, Any]` | Extensible audit/debug metadata. |

**Helper methods:**

- `score.task_count()` -- returns the number of tasks.
- `score.dependency_count()` -- returns the number of dependency edges.
- `score.validate()` -- validates structural consistency and returns an `EligibilityResult`.

### Task

A `Task` represents one pre-computed unit of work in the execution plan.

```python
from sonata import Task

task = Task(
    task_id=0,
    func_id=0,
    core_type="aic",
    args=("input_a", "input_b", "output_c"),
    arg_directions=("input", "input", "output"),
    arg_storage_keys=("param:input_a", "param:input_b", "alloc:output_c"),
    name="matmul_v2",
)
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `int` | Unique non-negative integer identifier within the score. |
| `func_id` | `int` | Non-negative integer identifying the callee function. |
| `core_type` | `str` | Target core type: `"aic"`, `"aiv"`, or `"mixed"`. |
| `args` | `tuple[Any, ...]` | Positional argument identifiers. |
| `arg_directions` | `tuple[str, ...]` | Direction for each arg: `"input"`, `"output"`, `"inout"`, `"scalar"`, `"nodep"`. |
| `arg_storage_keys` | `tuple[Any | None, ...]` | Storage identity key per arg, or `None` when unknown. |
| `name` | `str | None` | Optional callee function name. |

When `arg_directions` or `arg_storage_keys` are provided, their lengths must
match the length of `args`.

### Dependency

A `Dependency` is a directed edge between two tasks, expressing a scheduling
constraint.

```python
from sonata import Dependency

# RAW (read-after-write) data dependency
dep = Dependency(producer=0, consumer=1, kind="data")

# WAW (write-after-write) storage conflict
storage_dep = Dependency(producer=0, consumer=2, kind="storage")

# WAR (write-after-read) anti-dependency
war_dep = Dependency(producer=1, consumer=2, kind="war")

# Pure ordering constraint (no data flow)
order_dep = Dependency(producer=0, consumer=1, kind="ordering")
```

**Dependency kinds:**

| Kind | Semantics | Meaning |
|------|-----------|---------|
| `"data"` | RAW | Consumer reads data written by producer. |
| `"storage"` | WAW | Consumer writes to same storage as producer. |
| `"war"` | WAR | Consumer writes to storage previously read by producer. |
| `"ordering"` | Ordering | Pure sequencing constraint, no data flow. |

Dependency graphs must be acyclic. The `Score.validate()` method detects cycles
and reports them as validation errors.

### ShapeAssumption

A `ShapeAssumption` records a static shape fact that constrains the runtime
validity domain of a score. A score is only valid at runtime when the actual
tensor shapes match the assumed shapes.

```python
from sonata import ShapeAssumption

shape = ShapeAssumption(symbol="weight_matrix", dims=(512, 1024))
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | `str` | Name of the tensor or parameter. Must not be empty; must be unique within a score. |
| `dims` | `tuple[int, ...]` | Concrete positive integer dimensions. All values must be > 0. |

### RuntimeTarget

A `RuntimeTarget` carries runtime metadata for code generation.

```python
from sonata import RuntimeTarget

target = RuntimeTarget(
    runtime="host_build_graph",
    function_name="build_my_kernel_graph",
    aicpu_thread_num=4,
    config_comment=(
        "# Runtime configuration for host_build_graph.",
    ),
)
```

> **Note:** `Score.runtime_target` is deprecated since v0.2. Use
> `PlanHandle.runtime_target` instead. The `PlanHandle` is the proper carrier
> for runtime-specific metadata.

## Usage

### Constructing a Score

```python
from sonata import Score, RuntimeTarget, Task, Dependency, ShapeAssumption

score = Score(
    name="attention_layer",
    runtime_target=RuntimeTarget(
        runtime="host_build_graph",
        function_name="build_attention_layer_graph",
    ),
    tasks=(
        Task(task_id=0, func_id=0, core_type="aic", name="qkv_proj",
             args=("input", "weight", "output"),
             arg_directions=("input", "input", "output")),
        Task(task_id=1, func_id=1, core_type="aic", name="attention",
             args=("qkv", "attn_out"),
             arg_directions=("input", "output")),
        Task(task_id=2, func_id=2, core_type="aiv", name="feedforward",
             args=("attn_out", "ff_out"),
             arg_directions=("input", "output")),
    ),
    dependencies=(
        Dependency(producer=0, consumer=1, kind="data"),
        Dependency(producer=1, consumer=2, kind="data"),
    ),
    shape_assumptions=(
        ShapeAssumption(symbol="input", dims=(8, 128, 768)),
        ShapeAssumption(symbol="weight", dims=(768, 2304)),
    ),
)

print(f"Tasks: {score.task_count()}")
print(f"Edges: {score.dependency_count()}")
```

### Validating a Score

The `validate()` method checks structural consistency:

- Name is not empty.
- Task IDs are unique and non-negative.
- Function IDs are non-negative.
- Core types are `"aic"`, `"aiv"`, or `"mixed"`.
- `arg_directions` and `arg_storage_keys` lengths match `args` lengths.
- Dependency producer/consumer IDs reference existing tasks.
- No self-edges in dependencies.
- Dependency graph is acyclic.
- Shape assumption symbols are non-empty, unique, and have positive integer dims.

```python
result = score.validate()

if result.eligible:
    print("Score is valid.")
else:
    for reason in result.reasons:
        print(f"Validation error: {reason}")
```

The return type is `EligibilityResult`, which carries structured
`FallbackReason` entries with stable codes when validation fails.

### Dependency Policies

Sonata supports two built-in dependency construction policies:

- **`sequential_v0`** (default): Chains tasks in extraction order. Conservative
  and always correct, but limits parallelism.
- **`dataflow_v0`**: Builds RAW/WAW/WAR edges from task arg directions and
  storage keys. Enables more parallelism but requires complete direction
  information on every task.

```python
from sonata import build_dependencies, DEPENDENCY_POLICY_DATAFLOW_V0

edges = build_dependencies(score.tasks, policy=DEPENDENCY_POLICY_DATAFLOW_V0)
```

If dataflow analysis is not possible (missing directions), the system falls back
to `sequential_v0` and records the fallback reason in score metadata.

## Limitations and Known Constraints

- **Static shapes only.** Scores assume fixed tensor dimensions. Dynamic shapes
  are not supported; a score is invalid if shapes change at runtime.
- **No control flow.** The Score model represents straight-line task sequences.
  Loops and conditionals are handled at the region level (see
  [Regions and Multi-Adapter](regions-adapters.md)).
- **`runtime_target` on Score is deprecated.** Since v0.2, runtime metadata
  belongs on `PlanHandle`. Accessing `Score.runtime_target` emits a
  `DeprecationWarning`.
- **Frozen dataclasses.** All model types (`Score`, `Task`, `Dependency`, etc.)
  are frozen. Create new instances rather than mutating fields.
- **Dependency graph must be a DAG.** Cyclic dependencies are rejected by
  `validate()`.
