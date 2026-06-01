# Guard Condition Migration Guide

**Version**: v0.10  
**Status**: Complete  
**Author**: Sonata-Test (Qoder CLI CN)  

---

## Overview

This guide helps you migrate from `ShapeAssumption`-only systems to the unified `GuardCondition` abstraction introduced in v0.10.

### Why Migrate?

The new `GuardCondition` system provides several key benefits over the simple `ShapeAssumption` model:

1. **Unified Interface**: All guard types implement a common `GuardCondition` ABC
2. **Severity Classification**: Guards are classified as `soft` or `hard`, enabling smarter invalidation strategies
3. **Extensibility**: New guard types (e.g., `TopologyGuard`, `StorageGuard`) can be added without modifying existing code
4. **Over-Guarding Mitigation**: Built-in support for Top-K selection and density monitoring

### When to Migrate

- **v0.10**: Start using `GuardCondition` for new code; `ShapeAssumption` still works but emits deprecation warnings
- **v0.11**: `ShapeAssumption` will be removed; full migration required

---

## Quick Start

### Before (v0.9 and earlier)

```python
from sonata.score import ShapeAssumption

# Create shape guards
guards = (
    ShapeAssumption(symbol="batch_size", dims=(32,)),
    ShapeAssumption(symbol="seq_len", dims=(128,)),
)
```

### After (v0.10+)

```python
from sonata.guard import GuardCondition, GuardSeverity, GUARD_SEVERITY_HARD, GUARD_SEVERITY_SOFT
from sonata.score import ShapeAssumption  # Still works, but deprecated

# Option 1: Use ShapeAssumption with explicit severity (backward compatible)
guards = (
    ShapeAssumption(symbol="batch_size", dims=(32,), severity=GUARD_SEVERITY_HARD),
    ShapeAssumption(symbol="seq_len", dims=(128,), severity=GUARD_SEVERITY_HARD),
)

# Option 2: Create custom guard subclasses (recommended for new code)
class CustomGuard(GuardCondition):
    def __init__(self, symbol: str, threshold: int):
        super().__init__(symbol=symbol, severity=GUARD_SEVERITY_SOFT)
        self.threshold = threshold
    
    def evaluate(self, runtime_values: dict) -> bool:
        return runtime_values.get(self.symbol, 0) <= self.threshold

custom_guards = (
    CustomGuard("temp_limit", 80),
    CustomGuard("memory_usage", 90),
)
```

---

## Detailed Migration Steps

### Step 1: Update Imports

**Old**:
```python
from sonata.score import ShapeAssumption
```

**New** (recommended):
```python
from sonata.guard import (
    GuardCondition,
    GuardSeverity,
    GUARD_SEVERITY_HARD,
    GUARD_SEVERITY_SOFT,
)
from sonata.score import ShapeAssumption  # For backward compatibility
```

### Step 2: Add Severity to Existing ShapeAssumptions

**Old**:
```python
shape_assumptions = (
    ShapeAssumption(symbol="N", dims=(128, 64)),
    ShapeAssumption(symbol="M", dims=(256,)),
)
```

**New**:
```python
shape_assumptions = (
    ShapeAssumption(
        symbol="N", 
        dims=(128, 64), 
        severity=GUARD_SEVERITY_HARD  # Add severity
    ),
    ShapeAssumption(
        symbol="M", 
        dims=(256,), 
        severity=GUARD_SEVERITY_HARD
    ),
)
```

### Step 3: Choose Appropriate Severity

Use `GUARD_SEVERITY_HARD` for structural changes that require full replanning:

```python
# HARD: Shape changes → requires full replan
ShapeAssumption(symbol="batch_size", dims=(32,), severity=GUARD_SEVERITY_HARD)
ShapeAssumption(symbol="seq_len", dims=(128,), severity=GUARD_SEVERITY_HARD)
```

Use `GUARD_SEVERITY_SOFT` for parameter-level changes that can be handled by plan handle updates:

```python
# SOFT: Runtime parameters → can update plan handle in-place
CustomGuard(symbol="thread_count", value=4, severity=GUARD_SEVERITY_SOFT)
```

### Step 4: Create Custom Guard Subclasses (Optional)

For advanced use cases, create your own guard types:

```python
from sonata.guard import GuardCondition, GuardSeverity

class TopologyGuard(GuardCondition):
    """Guard for graph topology constraints."""
    
    def __init__(self, max_nodes: int):
        super().__init__(
            symbol="max_graph_nodes",
            severity=GuardSeverity("hard")  # Custom severity instance
        )
        self.max_nodes = max_nodes
    
    def evaluate(self, runtime_context: dict) -> bool:
        current_nodes = runtime_context.get("graph_node_count", 0)
        return current_nodes <= self.max_nodes


class MemoryGuard(GuardCondition):
    """Guard for memory usage constraints."""
    
    def __init__(self, max_memory_mb: int):
        super().__init__(
            symbol="max_memory_mb",
            severity=GuardSeverity("soft")
        )
        self.max_memory_mb = max_memory_mb
    
    def evaluate(self, runtime_context: dict) -> bool:
        current_memory = runtime_context.get("memory_used_mb", 0)
        return current_memory <= self.max_memory_mb
```

---

## API Reference

### Core Types

#### `GuardCondition` (ABC)

Abstract base class for all guard conditions.

```python
@dataclass(frozen=True)
class GuardCondition(ABC):
    symbol: str                    # Unique identifier for the guard
    severity: GuardSeverity        # Classification of invalidation severity
    
    @abstractmethod
    def evaluate(self, runtime_values: dict) -> bool:
        """Evaluate guard condition at runtime."""
        pass
```

#### `GuardSeverity`

Classification of guard invalidation severity.

```python
GUARD_SEVERITY_HARD = GuardSeverity("hard")   # Requires full replan
GUARD_SEVERITY_SOFT = GuardSeverity("soft")   # Can update plan handle in-place
```

#### `ShapeAssumption` (Deprecated)

Still works in v0.10, but emits deprecation warnings.

```python
@dataclass(frozen=True)
class ShapeAssumption(GuardCondition):
    symbol: str
    dims: tuple[int, ...]
    severity: GuardSeverity = GUARD_SEVERITY_HARD  # Default is HARD
```

### Migration Helpers

#### `shape_assumption_to_guard_condition()`

Converts `ShapeAssumption` to `GuardCondition` (with deprecation warning).

```python
from sonata.guard import shape_assumption_to_guard_condition

sa = ShapeAssumption(symbol="N", dims=(128,))
gc = shape_assumption_to_guard_condition(sa)  # Emits DeprecationWarning
```

#### `deprecated_shape_assumption()`

Helper function that creates `ShapeAssumption` with clear migration guidance.

```python
from sonata.guard import deprecated_shape_assumption

sa = deprecated_shape_assumption("N", (128,))  # Emits detailed deprecation warning
```

---

## Common Patterns

### Pattern 1: Static Shape Guards

Most common pattern - guards for tensor shapes:

```python
from sonata.guard import GUARD_SEVERITY_HARD
from sonata.score import ShapeAssumption

static_guards = (
    ShapeAssumption(symbol="batch_size", dims=(-1,), severity=GUARD_SEVERITY_HARD),
    ShapeAssumption(symbol="seq_len", dims=(512,), severity=GUARD_SEVERITY_HARD),
)
```

### Pattern 2: Dynamic Parameter Guards

Guards for runtime parameters that can change:

```python
from sonata.guard import GuardCondition, GUARD_SEVERITY_SOFT

class ThreadCountGuard(GuardCondition):
    def __init__(self, max_threads: int):
        super().__init__("max_threads", GUARD_SEVERITY_SOFT)
        self.max_threads = max_threads
    
    def evaluate(self, context: dict) -> bool:
        return context.get("thread_count", 1) <= self.max_threads
```

### Pattern 3: Combined Guards

Mix different guard types:

```python
from sonata.guard import GUARD_SEVERITY_HARD, GUARD_SEVERITY_SOFT
from sonata.score import ShapeAssumption

combined_guards = (
    # Structural guards (HARD)
    ShapeAssumption(symbol="input_shape", dims=(32, 224, 224, 3), severity=GUARD_SEVERITY_HARD),
    
    # Parameter guards (SOFT)
    ThreadCountGuard(max_threads=8),
    MemoryGuard(max_memory_mb=16384),
)
```

---

## Cache Integration

Guard conditions are automatically integrated into Score cache:

```python
from sonata.cache import ScoreCache, GuardStatus

cache = ScoreCache()
score = Score(...)  # Contains shape_assumptions

# Store with automatic guard evaluation
fp = cache.store(score)

# Lookup automatically checks guard status
payload = cache.lookup(fp)  # Returns None if guards violated

# Manual guard status update (advanced)
from sonata.plan_handle import PlanHandle
plan = PlanHandle.from_score(score)
cache.store_plan_handle(plan)  # Updates guard_status in cache
```

### Guard Status Values

- `ALL_SATISFIED`: All guards passed → cache hit
- `PARTIAL_FAILED`: Some guards failed → cache miss
- `ALL_FAILED`: All guards failed → cache miss

---

## Performance Considerations

### Benchmark Results

Guard checking overhead is minimal (< 5% for typical violation rates):

```bash
$ cd pypto-sonata && PYTHONPATH=src python benchmarks/bench_cache_guard_overhead.py

Baseline (ALL_SATISFIED):      3.5 µs per lookup
ALL_FAILED (all violated):     4.5 µs per lookup (+27%)
PARTIAL_FAILED (some violated): 3.5 µs per lookup (+0.3%)
```

### Recommendations

1. **Enable guard checking by default**: Overhead is negligible for normal workloads
2. **Monitor violation rates**: If > 50%, consider adaptive strategies
3. **Use appropriate severity**: Don't mark everything as HARD unnecessarily

---

## Troubleshooting

### Issue 1: "ShapeAssumption is deprecated" Warning

**Cause**: Using `ShapeAssumption` directly in v0.10+

**Solution**: Add explicit severity parameter:

```python
# Before (emits warning)
ShapeAssumption(symbol="N", dims=(128,))

# After (no warning)
ShapeAssumption(symbol="N", dims=(128,), severity=GUARD_SEVERITY_HARD)
```

### Issue 2: "Expected ShapeAssumption, got X" Error

**Cause**: Passing wrong type to `shape_assumption_to_guard_condition()`

**Solution**: Ensure you're passing a `ShapeAssumption` instance:

```python
from sonata.score import ShapeAssumption

sa = ShapeAssumption(symbol="N", dims=(128,))
gc = shape_assumption_to_guard_condition(sa)  # Works
```

### Issue 3: Cache Misses After Guard Violation

**Cause**: Guard status changed from `ALL_SATISFIED` to `PARTIAL_FAILED` or `ALL_FAILED`

**Solution**: This is expected behavior! The cache correctly invalidates when guards are violated. Rebuild the Score with updated assumptions.

---

## Migration Checklist

- [ ] Update imports to include `GuardCondition`, `GuardSeverity`
- [ ] Add `severity` parameter to all `ShapeAssumption` instances
- [ ] Replace hardcoded `GuardSeverity.HARD` with `GUARD_SEVERITY_HARD` constant
- [ ] Create custom guard subclasses for non-shape constraints
- [ ] Run compatibility tests to verify no regressions
- [ ] Update documentation with new API usage examples
- [ ] Plan for v0.11 removal of `ShapeAssumption`

---

## FAQ

**Q: Do I need to migrate immediately?**  
A: No. `ShapeAssumption` works in v0.10 with deprecation warnings. Full migration required in v0.11.

**Q: What's the difference between soft and hard guards?**  
A: Hard guards require full replanning (structural changes). Soft guards allow plan handle updates (parameter changes).

**Q: Can I mix old and new code during migration?**  
A: Yes. The migration utilities provide backward compatibility.

**Q: How do I know which guards should be soft vs hard?**  
A: Use HARD for shape/topology changes, SOFT for runtime parameters. When in doubt, start with HARD (conservative).

**Q: Will this break my existing code?**  
A: No. All changes are backward compatible. Deprecation warnings encourage migration but don't break functionality.

---

## Next Steps

1. **Start migration now**: Add severity to existing `ShapeAssumption` instances
2. **Plan for v0.11**: Schedule full migration before upgrading
3. **Explore advanced features**: Create custom guard subclasses for your use case
4. **Monitor performance**: Use benchmark tools to validate guard checking overhead

---

## References

- [`reports/v0.10-plan.md`](../../reports/v0.10-plan.md): v0.10 feature specification
- [`reports/roadmap_history.md`](../../reports/roadmap_history.md): Design decisions and rationale
- [`pypto-sonata/src/sonata/guard.py`](../../pypto-sonata/src/sonata/guard.py): Implementation source
- [`pypto-sonata/benchmarks/bench_cache_guard_overhead.py`](../../pypto-sonata/benchmarks/bench_cache_guard_overhead.py): Performance benchmarks

---

*Document created: 2026-06-01*  
*Last updated: 2026-06-01*  
*Version: v0.10*
