# Memory Planning

Sonata's memory planning system assigns non-overlapping memory offsets to buffers,
minimizing peak memory usage while respecting conflict constraints.

## Overview

```
Buffer lifetimes → Conflict matrix → Solver → MemoryPlan (offsets + sizes)
```

The pipeline has three stages:

1. **Liveness analysis** (`compute_lifetimes`) — determines when each buffer is alive
2. **Conflict matrix** (`compute_conflict_matrix`) — which buffers overlap in time
3. **Solver** (`GreedySolver` / custom `ConstraintSolver`) — assigns offsets

## Quick Start

```python
from sonata.liveness import BufferLifetime, compute_lifetimes
from sonata.memory_plan import compute_conflict_matrix, GreedySolver

# Define buffer lifetimes
lifetimes = [
    BufferLifetime(storage_key="input", birth=0, death=2),
    BufferLifetime(storage_key="hidden", birth=1, death=4),
    BufferLifetime(storage_key="output", birth=3, death=5),
]

# Build conflict matrix
matrix = compute_conflict_matrix(lifetimes)
# matrix[i][j] = True means buffers i and j overlap

# Solve for offsets
sizes = [1024, 2048, 512]
plan = GreedySolver().solve(matrix, sizes)

print(plan.peak_memory)  # total memory needed
for alloc in plan.allocations:
    print(f"{alloc.storage_key}: offset={alloc.offset}, size={alloc.size}")
```

## Conflict Matrix

`compute_conflict_matrix(lifetimes, stream_ids=None)` returns an NxN boolean matrix.

- `matrix[i][j] = True` — buffers i and j cannot share memory (lifetimes overlap)
- `matrix[i][i] = False` — no self-conflict by default
- Symmetric: `matrix[i][j] == matrix[j][i]`

### Stream-Aware Conflicts

Buffers on different hardware streams never conflict, even if lifetimes overlap:

```python
matrix = compute_conflict_matrix(
    lifetimes,
    stream_ids={"buf_a": 0, "buf_b": 1},  # different streams
)
```

## Solvers

### GreedySolver (default)

First-fit-decreasing by size. Sorts buffers largest-first, places each at the
first offset that doesn't conflict with already-placed buffers.

```python
from sonata.memory_plan import GreedySolver

solver = GreedySolver()
plan = solver.solve(matrix, sizes)
```

### Custom Solvers

Subclass `ConstraintSolver` for domain-specific strategies:

```python
from sonata.memory_plan import ConstraintSolver, MemoryPlan

class MySolver(ConstraintSolver):
    def solve(self, conflict_matrix, sizes, device_memory_limit=None):
        # Your algorithm here
        return MemoryPlan(allocations=..., peak_memory=...)
```

### Timeout and Fallback

`solve_memory()` runs a solver with timeout and automatic fallback:

```python
from sonata.memory_plan import solve_memory, GreedySolver

plan = solve_memory(
    primary_solver,
    matrix,
    sizes,
    timeout_seconds=1.0,    # fallback if primary exceeds this
    fallback=GreedySolver(), # default fallback
)
```

## MemoryPlan

The result contains:

- `allocations`: tuple of `BufferAllocation` (storage_key, offset, size)
- `peak_memory`: total memory used (highest offset + size)
- `solver_type`: which solver produced this plan (`"greedy"` or `"constraint_solver"`)
- `conflict_matrix_hash`: optional SHA-256 for cache key reuse

## Performance

Benchmarked on 50-buffer graphs with random lifetimes:
- **Conflict-matrix solver**: ~60% peak memory reduction vs naive sequential allocation
- **Solver time**: <1s for N≤100 buffers

## Integration with Region Analysis

Memory planning operates per-region in the v0.11+ region tree:
- Each static region gets independent memory planning
- Dynamic regions use fallback allocation
- Region guard status tracks whether plans remain valid
