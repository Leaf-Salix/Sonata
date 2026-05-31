# PyPTO Integration

## Overview

Sonata integrates with PyPTO through a well-defined adapter boundary. The
`PostSimplifyPyPTOInputAdapter` projects a certified post-Simplify PyPTO IR
shape into the small set of normalized facts that Sonata needs for Score
construction. This keeps Sonata's core scheduling logic decoupled from PyPTO's
C++ IR bindings while operating on a well-defined pipeline stage contract.

The adapter operates on the named certified dump produced after
`CollectCommGroups` and the final `Simplify` pass, before
`MaterializeRuntimeScopes`.

## Key Concepts

### PostSimplifyPyPTOInputAdapter

The adapter is the bridge between PyPTO IR and Sonata's pure-Python task model.
It uses structural Python-visible fields instead of importing PyPTO IR classes.

```python
from sonata.pypto_adapter import PostSimplifyPyPTOInputAdapter

adapter = PostSimplifyPyPTOInputAdapter(ir_node, entry_name="my_function")
```

**Constructor parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `node` | `Any` | The PyPTO IR node (typically a `Function` or `Program`). |
| `entry_name` | `str | None` | Optional name of the target orchestration function. |

### Certified Dump

The adapter targets a specific PyPTO pipeline stage called the **certified
dump**. This is a named checkpoint in the PyPTO compilation pipeline where the IR
has known structural properties that Sonata depends on.

```python
from sonata.pypto_adapter import (
    DEFAULT_CERTIFIED_DUMP,
    CERTIFIED_DUMP_AFTER_COLLECT_COMM_GROUPS_SIMPLIFY,
)

# The default certified dump stage name
print(DEFAULT_CERTIFIED_DUMP)
# "after_collect_comm_groups_simplify"

# This is the stage after CollectCommGroups and the final Simplify pass,
# but before MaterializeRuntimeScopes.
```

At the certified dump stage, the adapter expects:

- All orchestration functions are present and identifiable.
- Call nodes carry `arg_directions` matching their `args` in count.
- No runtime scope materialization has occurred yet.

### Normalized Facts

The adapter produces normalized fact objects that Sonata uses for Score
construction. These are pure Python dataclasses, decoupled from PyPTO IR types.

#### NormalizedTaskFacts

The top-level fact container:

```python
from sonata.pypto_adapter import NormalizedTaskFacts

# Fields:
# facts.source           -- the original IR node
# facts.certified_dump   -- the certified dump stage name
# facts.functions        -- tuple of NormalizedFunctionFact
# facts.dependency_facts -- reserved for future dependency provenance
```

#### NormalizedFunctionFact

Facts for one orchestration root function:

```python
from sonata.pypto_adapter import NormalizedFunctionFact

# Fields:
# fact.node             -- the IR function node
# fact.name             -- function name (str | None)
# fact.is_orchestration -- always True for selected roots
# fact.params           -- function parameters
# fact.calls            -- tuple of NormalizedCallFact
```

#### NormalizedCallFact

Facts for one ordinary orchestration call:

```python
from sonata.pypto_adapter import NormalizedCallFact

# Fields:
# fact.node             -- the IR Call node
# fact.callee_name      -- callee function name
# fact.args             -- raw call arguments
# fact.arg_names        -- string names for each argument
# fact.arg_directions   -- direction for each arg ("input", "output", etc.)
# fact.arg_storage_keys -- storage identity for each arg
# fact.core_type        -- target core type ("aic", "aiv", "mixed")
```

### Extraction Roots

Extraction roots are the orchestration functions that the adapter selects for
Score construction:

```python
adapter = PostSimplifyPyPTOInputAdapter(program_node)

roots = adapter.extraction_roots()
for root in roots:
    print(f"Root: {adapter.function_name(root)}")
    print(f"  Is orchestration: {adapter.is_orchestration(root)}")
    print(f"  Core type: {adapter.core_type_from_function(root)}")
```

**Selection rules:**

1. If the node has a `functions` dict (i.e., it is a `Program`), all
   `Orchestration` functions are selected.
2. If `entry_name` is provided, only functions matching that name are selected.
3. If the node is a direct `Function`, it is selected only if its
   `func_type.name` is `"Orchestration"`.

### IR Walking

The adapter provides a structural walker that traverses common IR child fields:

```python
adapter = PostSimplifyPyPTOInputAdapter(node)

for child in adapter.walk(node):
    kind = adapter.kind(child)
    if kind == "Call":
        call_name = adapter.call_name(child)
        if call_name and not adapter.is_builtin_call(call_name):
            print(f"User call: {call_name}")
```

The walker visits nodes through these child fields: `functions`, `body`,
`then_body`, `else_body`, `branches`, `stmts`, `statements`, `seq`, `args`,
`value`, `expr`, `condition`. It handles cycles by tracking visited node
identity.

### Builtin Operations

The adapter classifies certain call names as builtins that should not become
Score tasks:

```python
# Builtin prefixes: "tile.", "tensor.", "system.", "array."
PostSimplifyPyPTOInputAdapter.is_builtin_call("tensor.create")  # True
PostSimplifyPyPTOInputAdapter.is_builtin_call("tile.sync")       # True
PostSimplifyPyPTOInputAdapter.is_builtin_call("matmul_v2")       # False
```

## Usage

### Adapter Workflow

The typical workflow is:

1. Obtain the PyPTO IR at the certified dump stage.
2. Create the adapter.
3. Check for scope/contract errors.
4. Normalize into facts.
5. Use facts for Score construction (usually via `check_static_eligibility`).

```python
from sonata.pypto_adapter import (
    PostSimplifyPyPTOInputAdapter,
    PyPTOAdapterContractError,
)

# 1. Create the adapter
adapter = PostSimplifyPyPTOInputAdapter(ir_node, entry_name="main")

# 2. Check for scope violations
errors = adapter.out_of_scope_errors()
if errors:
    for err in errors:
        print(f"Scope error: {err}")
    # Fall back to dynamic runtime
    return

# 3. Normalize into facts
try:
    facts = adapter.normalize()
except PyPTOAdapterContractError as exc:
    print(f"Contract error: {exc}")
    return

# 4. Inspect the facts
for func in facts.functions:
    print(f"Function: {func.name}")
    for call in func.calls:
        print(f"  Call: {call.callee_name} ({call.core_type})")
        print(f"    Args: {call.arg_names}")
        print(f"    Directions: {call.arg_directions}")
```

### Certified Contract Validation

For stricter validation, use `require_certified=True` to enforce the full
certified dump contract:

```python
# Check certified contract errors before normalizing
errors = adapter.certified_contract_errors()
if errors:
    for err in errors:
        print(f"Certified contract violation: {err}")
    return

# Or let normalize() raise on contract errors
try:
    facts = adapter.normalize(require_certified=True)
except PyPTOAdapterContractError as exc:
    print(f"Contract violation: {exc}")
```

The certified contract additionally requires:

- Every call has `arg_directions` present.
- `arg_directions` length matches `args` length for every call.

### Integration with check_static_eligibility

In practice, `check_static_eligibility` wraps the adapter internally:

```python
from sonata import check_static_eligibility

# This internally creates a PostSimplifyPyPTOInputAdapter,
# checks scope errors, normalizes, and builds a Score
result = check_static_eligibility(
    ir_node,
    entry_name="main",
    require_certified=True,
)

if result.eligible:
    score = result.score
```

### Function Type and Core Type Detection

```python
adapter = PostSimplifyPyPTOInputAdapter(program_node)

for func in adapter.extraction_roots():
    func_type = adapter.function_type_name(func)   # "Orchestration", "AIC", "AIV", ...
    core_type = adapter.core_type_from_function(func)  # "aic", "aiv", "mixed"
    name = adapter.function_name(func)

    print(f"{name}: type={func_type}, core={core_type}")
```

Core type mapping:

| `func_type.name` | Sonata core type |
|------------------|-----------------|
| `"AIC"` | `"aic"` |
| `"AIV"` | `"aiv"` |
| anything else | `"mixed"` |

### Argument Direction Extraction

The adapter extracts argument directions from Call nodes:

```python
directions = adapter.arg_directions(call_node)
# Returns a tuple like ("input", "input", "output")

# Individual direction normalization
name = adapter.direction_name(direction_enum)
# Handles enum .name, .value, or str conversion
```

## Scope Errors and Contract Errors

The adapter distinguishes two error categories:

**Scope errors** (`out_of_scope_errors()`): Structural violations that mean
the IR is out of scope for Sonata v0.1.

- Non-orchestration function root.
- `Group`/`Spmd` callee called from an orchestration root.
- `RuntimeScopeStmt`, `Submit`, `SpmdScopeStmt`, or `ManualScopeStmt` in the
  IR tree.

**Certified contract errors** (`certified_contract_errors()`): Includes all
scope errors plus:

- Call nodes missing `arg_directions`.
- `arg_directions` size not matching `args` size.

## Limitations and Known Constraints

- **Single pipeline stage.** The adapter targets one specific certified dump
  stage (`after_collect_comm_groups_simplify`). If a different stage boundary
  is needed, create a separate adapter class rather than generalizing this one.
- **Structural introspection only.** The adapter uses Python-visible fields
  (`name`, `func_type`, `body`, `args`, etc.) rather than importing PyPTO C++
  classes. This keeps coupling low but means it depends on field naming
  conventions.
- **No Group/Spmd support.** Functions with `func_type.name` of `"Group"` or
  `"Spmd"` are out of scope. Calls to Group/Spmd functions from orchestration
  roots are rejected.
- **No control flow.** The adapter does not process `ForStmt`, `IfStmt`, or
  `WhileStmt`. These are handled at the region level.
- **Builtin filtering is prefix-based.** Calls starting with `"tile."`,
  `"tensor."`, `"system."`, or `"array."` are treated as builtins and excluded
  from Score tasks. Custom operations with these prefixes would be
  misclassified.
- **`PyPTOAdapterContractError`** is raised (not returned) when `normalize()`
  encounters scope or contract violations. Callers must catch the exception.
