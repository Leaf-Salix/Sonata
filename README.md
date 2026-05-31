# Sonata

Sonata is a host-side static scheduling compiler layer for PyPTO. For
static-shape subgraphs it generates a pre-computed task dependency calendar
(Score) at compile time, consumed by the `host_build_graph` runtime instead of
AICPU dynamic scheduling. Sonata extracts structural facts from the PyPTO
IR without importing C++ bindings, producing inspectable JSON-serializable
plans that can be cached, validated, and round-tripped through serialization.

**Version**: v0.9 (pre-release)

## Architecture

```text
                     PyPTO IR / DSL
                           |
                  +--------v---------+
                  |  PyPTO Adapter   |  Structural extraction
                  | (post_simplify)  |  (no C++ import)
                  +--------+---------+
                           |
                  +--------v---------+
                  |   Eligibility    |  Static shape / control-flow
                  |     Check        |  / storage coverage guards
                  +--------+---------+
                           |
              +------------+-------------+
              | eligible?                | rejected
              v                          v
     +--------+---------+     +----------+---------+
     |  Score Builder   |     |  FallbackReason    |
     |  - Tasks         |     |  (structured codes)|
     |  - Dependencies  |     +--------------------+
     |  - Shape Assumptions
     |  - Storage Keys  |
     +--------+---------+
              |
     +--------v---------+
     |  PlanHandle      |  Runtime artifact key:
     |  - fingerprint   |  score_fingerprint,
     |  - func registry |  RuntimeTarget,
     |  - arg bindings  |  source adapter
     +--------+---------+
              |
     +--------v---------+
     | Runtime Adapter  |  Score + PlanHandle ->
     | (HostBuildGraph) |  task table + edge table
     +------------------+
              |
              v
     Host-side graph runtime
```

Supporting subsystems:

- **Score Cache** -- fingerprint-keyed in-memory and persistent cache that
  avoids repeating eligibility checks for the same static subgraph.
- **Region Extraction** -- splits IR into static and dynamic regions so that
  only static portions are planned by Sonata while dynamic regions fall back
  to the original PyPTO runtime.
- **Alias Analysis** -- determines alias / view / inplace / disjoint
  relationships between storage keys.
- **Liveness & Memory Planning** -- computes buffer lifetime intervals and
  produces a greedy-first-fit memory layout plan.

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Run tests

Pure Sonata tests (no PyPTO dependency):

```bash
PYTHONPATH=src python -m pytest tests/sonata
```

With a PyPTO checkout or submodule:

```bash
PYTHONPATH=src:upstream/pypto/python python -m pytest tests/sonata tests/pypto_cases
```

### Basic usage

```python
from sonata import check_static_eligibility, score_to_json, score_fingerprint

# Check if a PyPTO IR node is eligible for static planning
result = check_static_eligibility(ir_node, entry_name="my_orchestration")

if result.eligible:
    score = result.score
    print(f"Planned {score.task_count()} tasks, "
          f"{score.dependency_count()} dependencies")
    print(f"Fingerprint: {score_fingerprint(score)[:16]}...")
    print(score_to_json(score))
else:
    for detail in result.reason_details:
        print(f"[{detail.code}] {detail.message}")
```

Building a PlanHandle from a Score:

```python
from sonata import PlanHandle, HostBuildGraphRuntimeAdapter

handle = PlanHandle.from_score(score, source_adapter="post_simplify")

adapter = HostBuildGraphRuntimeAdapter()
adapter_result = adapter.generate(score, handle)
if adapter_result.success:
    plan = adapter_result.plan
    print(f"{plan.task_count()} tasks, {plan.edge_count()} edges")
```

Using region-level eligibility:

```python
from sonata import extract_regions, check_region_eligibility

region_map = extract_regions(ir_node)
print(f"Static: {len(region_map.static_regions())}, "
      f"Dynamic: {len(region_map.dynamic_regions())}")

result = check_region_eligibility(ir_node)
```

Using the Score cache:

```python
from sonata import ScoreCache, cached_score

cache = ScoreCache()
score, fingerprint, was_cached = cached_score(
    cache, lambda: build_score(), fingerprint_hint=hint,
)
cache.save("score_cache.json")
```

## Key Concepts

### Score

The central data model. A `Score` is a frozen, inspectable static execution
plan emitted before target-specific codegen. It carries:

- **Tasks** -- precomputed runtime tasks with func_id, core_type, args,
  arg_directions, and arg_storage_keys.
- **Dependencies** -- explicit edges between tasks, classified by kind:
  `data` (RAW), `storage` (WAW), `war` (WAR), or `ordering`.
- **Shape Assumptions** -- the static shape facts that define the runtime
  validity domain.
- **Metadata** -- audit data including storage coverage, dependency policy,
  and extraction provenance.

### Eligibility

`check_static_eligibility()` performs conservative static checks on a PyPTO
IR node. It returns an `EligibilityResult` that is either:

- **eligible** -- carrying a Score, or
- **rejected** -- carrying structured `FallbackReason` entries with stable
  `FallbackCode` enum values.

Eligibility covers root kind validation, control-flow detection, storage
coverage thresholds, dataflow direction completeness, and score self-consistency
(acyclic dependencies, unique task IDs, valid shape assumptions).

### Fingerprint

`score_fingerprint()` produces a stable SHA-256 digest of a Score's
computation identity (tasks, dependencies, shape assumptions). Fingerprints
serve as cache keys and PlanHandle binding tokens. The fingerprint excludes
runtime-specific metadata so the same computation yields the same digest
regardless of target.

### PlanHandle

`PlanHandle` bridges a Score (computation identity) to a specific runtime
target. It carries:

- `score_fingerprint` -- binds to the originating Score.
- `runtime_target` -- the runtime configuration (e.g., host_build_graph).
- `func_registry` -- maps Sonata func_ids to runtime codegen func_ids.
- `arg_bindings` -- maps task args from Sonata storage identity to runtime
  handles.
- `runtime_contract_version` -- for forward-compatible contract evolution.

### Dependency Policies

Two policies are available:

- `sequential_v0` -- chains tasks in extraction order (safe default).
- `dataflow_v0` -- builds conservative RAW/WAW/WAR edges from task arg
  directions and storage keys. Falls back to `sequential_v0` when direction
  data is incomplete.

### Serialization

Scores, PlanHandles, and EligibilityResults support bidirectional
serialization to deterministic JSON-compatible dictionaries with schema
versioning for forward compatibility.

## Project Layout

```text
src/sonata/          # Sonata package source
tests/sonata/        # PyPTO-free unit tests
tests/pypto_cases/   # PyPTO compatibility and extraction tests
patches/pypto/       # Minimal PyPTO seam patches
docs/                # Project documentation
upstream/pypto/      # PyPTO submodule (future)
```

## Requirements

- Python 3.10+
- pytest >= 7.0 (for development)
- No runtime dependencies

## License

CANN Open Software License Agreement Version 2.0

See [LICENSE](LICENSE) for the full text.
