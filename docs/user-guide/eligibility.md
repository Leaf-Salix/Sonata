# Eligibility Checking

## Overview

Before Sonata can generate a static execution plan, it must determine whether a
given IR graph (or region of one) is eligible for static planning. The
eligibility module provides conservative structural checks that explain *why* a
graph cannot be statically planned, producing structured fallback reasons that
callers can use to route back to the original PyPTO dynamic runtime path.

The two main entry points are:

- `check_static_eligibility()` -- whole-graph eligibility.
- `check_region_eligibility()` -- per-region eligibility (see
  [Regions and Multi-Adapter](regions-adapters.md)).

## Key Concepts

### EligibilityResult

Every eligibility check returns an `EligibilityResult`, which is the unified
answer to "can Sonata plan this?"

```python
from sonata import EligibilityResult

# Inspecting the result
result = check_static_eligibility(ir_node)

if result.eligible:
    score = result.score  # the constructed Score
    print(f"Eligible! Score has {score.task_count()} tasks.")
else:
    for detail in result.reason_details:
        print(f"[{detail.code}] {detail.message} (severity: {detail.severity})")
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `eligible` | `bool` | Whether the input is eligible for Sonata planning. |
| `score` | `Score | None` | The constructed Score when eligible; `None` otherwise. |
| `reasons` | `tuple[str, ...]` | Human-readable reason messages. |
| `reason_details` | `tuple[FallbackReason, ...]` | Structured reason entries with stable codes. |

**Factory methods:**

```python
# Build an eligible result
result = EligibilityResult.accept(score)

# Build an ineligible result
result = EligibilityResult.reject("control flow not supported", fallback_reason)

# Build an eligible result with degraded-confidence warnings
result = EligibilityResult.accept_with_warnings(score, warning_reason)
```

**Query methods:**

- `result.has_errors()` -- returns `True` if any detail has severity `"error"`.
- `result.has_warnings()` -- returns `True` if any detail has severity `"warning"`.

### FallbackReason

A `FallbackReason` is a structured explanation for why a score or region is
ineligible. It carries a stable code, a human-readable message, and a severity.

```python
from sonata import FallbackReason

reason = FallbackReason(
    code="control_flow_not_supported",
    message="IfStmt is not supported by initial Sonata eligibility",
    severity="error",
)
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `code` | `str` | Stable routing key. Enum codes from `FallbackCode` are guaranteed stable. Slug codes derived from messages may change. |
| `message` | `str` | Human-readable explanation. |
| `severity` | `str` | `"error"` for rejections, `"warning"` for degraded-confidence acceptances. |

### FallbackCode

`FallbackCode` is an enum of stable reason codes. Use these as routing or
filtering keys -- they do not change when message wording changes.

```python
from sonata import FallbackCode

# Common codes
FallbackCode.UNSUPPORTED_ROOT_KIND          # Root IR node type not supported
FallbackCode.CONTROL_FLOW_NOT_SUPPORTED      # ForStmt/IfStmt/WhileStmt found
FallbackCode.UNSUPPORTED_RUNTIME_SCOPE       # RuntimeScopeStmt found
FallbackCode.TENSOR_READ_NOT_SUPPORTED       # tensor.read call found
FallbackCode.ENTRY_FUNCTION_NOT_ORCHESTRATION # Entry is not an Orchestration function
FallbackCode.UNSUPPORTED_PYPTO_ADAPTER_SCOPE # IR out of scope for v0.1 adapter
FallbackCode.SCORE_VALIDATION_FAILED         # Score internal validation failed
FallbackCode.STORAGE_COVERAGE_BELOW_THRESHOLD # Storage key coverage too low
FallbackCode.DATAFLOW_DIRECTIONS_UNAVAILABLE # No arg directions for dataflow
FallbackCode.DATAFLOW_DIRECTIONS_INCOMPLETE  # Partial arg directions
```

Runtime adapter codes (used by `HostBuildGraphRuntimeAdapter`):

```python
FallbackCode.RUNTIME_ADAPTER_FINGERPRINT_MISMATCH
FallbackCode.RUNTIME_ADAPTER_CONTRACT_VERSION_MISMATCH
FallbackCode.RUNTIME_ADAPTER_FUNC_NOT_REGISTERED
FallbackCode.RUNTIME_ADAPTER_FUNC_UNREFERENCED
FallbackCode.RUNTIME_ADAPTER_BINDING_INCOMPLETE
FallbackCode.RUNTIME_ADAPTER_INVALID_EDGE
```

## Usage

### check_static_eligibility

The primary entry point for whole-graph eligibility checking:

```python
from sonata import check_static_eligibility

result = check_static_eligibility(
    ir_node,
    entry_name="my_orchestration",       # optional: target a specific function
    dependency_policy="sequential_v0",    # or "dataflow_v0"
    require_certified=False,              # set True for certified dump validation
)

if result.eligible:
    score = result.score
    # Proceed with serialization, caching, or runtime plan generation
else:
    # Fall back to dynamic runtime
    for detail in result.reason_details:
        if detail.code == "control_flow_not_supported":
            print("Graph contains control flow; use region-level eligibility.")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `node` | `Any` | (required) | The IR node to check (typically a `Function` or `Program`). |
| `runtime_target` | `RuntimeTarget | None` | `None` | Optional runtime target override. |
| `entry_name` | `str | None` | `None` | Name of the orchestration function to target. |
| `dependency_policy` | `str` | `"sequential_v0"` | Dependency construction policy. |
| `require_certified` | `bool` | `False` | When `True`, enforce certified dump contract checks. |

### What Makes an IR Eligible

An IR graph is eligible for static planning when:

1. **Root kind is supported.** The root node must be a `Function` or `Program`.
2. **No control flow.** The graph must not contain `ForStmt`, `IfStmt`, or
   `WhileStmt` nodes.
3. **No runtime scopes.** `RuntimeScopeStmt` nodes are not supported.
4. **No tensor.read calls.** `tensor.read` operations are not supported.
5. **Entry function exists.** When `entry_name` is provided, at least one
   `Orchestration` function with that name must exist.
6. **Adapter scope.** The IR must be within scope of the PyPTO input adapter
   (no `Group`/`Spmd` callees, no unsupported scope statements).
7. **Score validation passes.** The constructed score must pass internal
   validation (unique task IDs, acyclic dependencies, valid shapes, etc.).

### Storage Coverage Warnings

After a score passes validation, eligibility also checks storage key coverage.
When the ratio of known storage keys to total memory-direction arguments falls
below the threshold (default 0.5), the result is still eligible but carries a
warning:

```python
result = check_static_eligibility(ir_node)

if result.eligible and result.has_warnings():
    for detail in result.reason_details:
        if detail.severity == "warning":
            print(f"Warning: {detail.message}")
```

### Dependency Policy Fallback

When `dependency_policy="dataflow_v0"` is requested but tasks lack complete
`arg_directions`, the system automatically falls back to `sequential_v0` and
records the reason:

```python
from sonata import check_static_eligibility, DEPENDENCY_POLICY_DATAFLOW_V0

result = check_static_eligibility(
    ir_node,
    dependency_policy=DEPENDENCY_POLICY_DATAFLOW_V0,
)

# Check metadata for fallback information
if result.eligible:
    meta = result.score.metadata
    if "dependency_policy_fallback_reason" in meta:
        print(f"Requested dataflow_v0 but fell back: {meta['dependency_policy_fallback_reason']}")
```

### Filtering by FallbackCode

Use `FallbackCode` values to route different rejection reasons:

```python
from sonata import FallbackCode

result = check_static_eligibility(ir_node)

if not result.eligible:
    codes = {detail.code for detail in result.reason_details}
    if FallbackCode.CONTROL_FLOW_NOT_SUPPORTED.value in codes:
        # Try region-level eligibility instead
        from sonata import check_region_eligibility
        result = check_region_eligibility(ir_node)
    elif FallbackCode.UNSUPPORTED_PYPTO_ADAPTER_SCOPE.value in codes:
        print("IR is out of scope for the current adapter version.")
```

## Limitations and Known Constraints

- **Structural checks only.** Eligibility uses Python-visible IR fields, not
  full semantic analysis. Some runtime-invalid graphs may pass eligibility.
- **Conservative by design.** When in doubt, eligibility rejects. False
  negatives (rejecting an eligible graph) are preferred over false positives.
- **Whole-graph is all-or-nothing.** `check_static_eligibility` requires the
  entire graph to be static. For mixed static/dynamic graphs, use
  `check_region_eligibility` for per-region granularity.
- **Certified dump contract.** When `require_certified=True`, the adapter
  enforces that all calls carry `arg_directions` matching the certified PyPTO
  dump stage. This catches stage-mismatch errors early.
- **Message wording is not an API contract.** Use `FallbackCode` enum values
  for routing; raw message strings may change between versions.
