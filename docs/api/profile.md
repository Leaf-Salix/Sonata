# Profile API

Operator execution profiling for feedback-driven scheduling (v0.18 Phase 3).

## `OperatorProfile`

```python
from sonata.profile import OperatorProfile

profile = OperatorProfile(
    op_signature="matmul_fp16_128x128x128",
    op_type="matmul",
    shape=(128, 128, 128),
    dtype="fp16",
    core_type="aic",
    mean_latency_us=320.0,
    std_latency_us=12.5,
    sample_count=150,
)
```

Frozen dataclass. Fields:

| Field | Type | Description |
|-------|------|-------------|
| `op_signature` | `str` | Unique key, e.g. `"matmul_fp16_128x128x128"` |
| `op_type` | `str` | Operator type |
| `shape` | `tuple[int, ...]` | Tensor shape |
| `dtype` | `str` | Data type |
| `core_type` | `str` | Core type (aic/aiv) |
| `mean_latency_us` | `float` | Mean execution time in microseconds |
| `std_latency_us` | `float` | Standard deviation |
| `sample_count` | `int` | Number of recorded samples |

## `ProfileDatabase`

```python
from sonata.profile import ProfileDatabase

db = ProfileDatabase()
db.record("matmul", (128, 128), "fp16", "aic", 320.0)
db.record("matmul", (128, 128), "fp16", "aic", 340.0)

profile = db.lookup("matmul", (128, 128), "fp16")
# profile.mean_latency_us ≈ 330.0, sample_count = 2

db.save("profiles.json")
db.load("profiles.json")
```

### Methods

| Method | Description |
|--------|-------------|
| `lookup(op_type, shape, dtype)` | Return profile or None |
| `record(op_type, shape, dtype, core_type, latency_us)` | Record one sample (incremental mean/std) |
| `all_profiles()` | Return all stored profiles |
| `save(path)` | Save to JSON |
| `load(path)` | Load from JSON (merges with existing) |

### Mean/Std Update

Uses Welford's online algorithm for incremental mean and variance:

```
new_mean = old_mean + (x - old_mean) / n
new_var = old_var + (x - old_mean) * (x - new_mean)
```

## Timing Collection

```python
from sonata.pipeline import collect_task_timings

collect_task_timings(db, score, {"matmul": 320.0, "add": 50.0})
```

Records per-task execution latency into the database after execution.

## Profile-Aware Scheduling

```python
from sonata.pipeline import compute_scheduling_instructions

# Without profile (default)
instructions = compute_scheduling_instructions(dispatch)

# With profile
instructions = compute_scheduling_instructions(dispatch, profile_db=db)
```

When profile data is available:
- High latency (>1ms) → higher block_dim for more parallelism
- Low latency (<100us) → lower block_dim to reduce overhead
