# Runtime Hook API

The runtime hook is Sonata's formal integration point with PyPTO's `execute_compiled()`. It is an **optional** module — PyPTO imports it via `try/except ImportError`, so Sonata is not required.

## `apply_sonata_runtime_hints()`

```python
from sonata.runtime_hook import apply_sonata_runtime_hints

hints = apply_sonata_runtime_hints(
    work_dir="/path/to/compiled",
    block_dim=None,           # current effective block_dim
    aicpu_thread_num=None,    # current effective thread count
    user_block_dim=None,      # original user arg (before RUNTIME_CONFIG fallback)
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `work_dir` | `str \| Path` | Compiled artifacts directory containing `sonata_plan.json` |
| `block_dim` | `int \| None` | Current effective block_dim (may come from RUNTIME_CONFIG) |
| `aicpu_thread_num` | `int \| None` | Current effective thread count |
| `user_block_dim` | `int \| None` | Original user-supplied block_dim. When not None, Sonata will not override |

### Returns: `SonataRuntimeHints`

| Field | Type | Description |
|-------|------|-------------|
| `block_dim` | `int \| None` | (Possibly updated) block_dim |
| `aicpu_thread_num` | `int \| None` | (Possibly updated) thread count |
| `sonata_applied` | `bool` | Whether Sonata actually modified the parameters |
| `reason` | `str` | Why Sonata did or didn't apply |

### Behavior

1. No `sonata_plan.json` → returns original params, `sonata_applied=False`
2. Plan not eligible → returns original params
3. `user_block_dim` is not None → returns original params (user takes precedence)
4. El eligible plan, no user block_dim → computes from `dispatch_regions()` + `compute_scheduling_instructions()`
5. Any error → fail-open: logs warning, returns original params

### Block Dim Rules

| Region Type | Default block_dim |
|-------------|-------------------|
| static | 32 (or profile-informed) |
| dynamic | 1 |
| mixed | 16 |

## Integration with PyPTO Runner

The hook is called in `upstream/pypto/python/pypto/runtime/runner.py`:

```python
try:
    from sonata.runtime_hook import apply_sonata_runtime_hints
except ImportError:
    pass
else:
    hints = apply_sonata_runtime_hints(
        work_dir=work_dir,
        block_dim=effective_block_dim,
        aicpu_thread_num=effective_aicpu_thread_num,
        user_block_dim=block_dim,  # before RUNTIME_CONFIG fallback
    )
    if hints.sonata_applied:
        effective_block_dim = hints.block_dim
        effective_aicpu_thread_num = hints.aicpu_thread_num
```
