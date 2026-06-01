"""Compatibility tests for v0.10 GuardCondition migration.

These tests verify that:
1. ShapeAssumption still works as a GuardCondition subclass
2. Deprecation warnings are emitted correctly
3. Migration utilities function properly
4. Backward compatibility is maintained

Run with:
    cd pypto-sonata && PYTHONPATH=src python -m pytest tests/sonata/test_migration_compatibility.py -v
"""

import pytest
import warnings

from sonata.guard import (
    GuardCondition,
    GuardSeverity,
    GUARD_SEVERITY_HARD,
    GUARD_SEVERITY_SOFT,
    shape_assumption_to_guard_condition,
    deprecated_shape_assumption,
)
from sonata.score import ShapeAssumption


class TestShapeAssumptionAsGuardCondition:
    """Verify ShapeAssumption remains a valid GuardCondition subclass."""
    
    def test_shape_assumption_is_guard_condition(self):
        """ShapeAssumption should be an instance of GuardCondition."""
        sa = ShapeAssumption(symbol="N", dims=(128,))
        assert isinstance(sa, GuardCondition)
    
    def test_shape_assumption_has_required_fields(self):
        """ShapeAssumption must have symbol and severity fields."""
        sa = ShapeAssumption(symbol="batch", dims=(32,))
        assert hasattr(sa, "symbol")
        assert hasattr(sa, "severity")
        assert sa.symbol == "batch"
        assert sa.severity in (GUARD_SEVERITY_HARD, GUARD_SEVERITY_SOFT)
    
    def test_shape_assumption_default_severity(self):
        """ShapeAssumption defaults to HARD severity."""
        sa = ShapeAssumption(symbol="N", dims=(128,))
        assert sa.severity == GUARD_SEVERITY_HARD
    
    def test_shape_assumption_custom_severity(self):
        """ShapeAssumption accepts custom severity."""
        sa_soft = ShapeAssumption(symbol="M", dims=(64,), severity=GUARD_SEVERITY_SOFT)
        assert sa_soft.severity == GUARD_SEVERITY_SOFT
        
        sa_hard = ShapeAssumption(symbol="K", dims=(32,), severity=GUARD_SEVERITY_HARD)
        assert sa_hard.severity == GUARD_SEVERITY_HARD


class TestDeprecationWarnings:
    """Verify deprecation warnings are emitted correctly."""
    
    def test_shape_assumption_direct_creation_emits_warning(self):
        """Direct ShapeAssumption creation emits DeprecationWarning per v0.10 plan."""
        # Per v0.10-plan.md section F2: "使用 ShapeAssumption 时能看到清晰的 deprecation warning"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sa = ShapeAssumption(symbol="N", dims=(128,))
            
            # Should have at least one ShapeAssumption-specific warning
            shape_warnings = [x for x in w if "ShapeAssumption" in str(x.message)]
            assert len(shape_warnings) >= 1
            
            # Last warning should be DeprecationWarning
            assert issubclass(shape_warnings[-1].category, DeprecationWarning)
            msg = str(shape_warnings[-1].message)
            assert "deprecated" in msg.lower()
            assert "v0.10" in msg or "v0.11" in msg
    
    def test_conversion_function_emits_warning(self):
        """shape_assumption_to_guard_condition should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sa = ShapeAssumption(symbol="N", dims=(128,))
            gc = shape_assumption_to_guard_condition(sa)
            
            assert len(w) >= 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert "migration" in str(w[-1].message).lower() or "deprecated" in str(w[-1].message).lower()
    
    def test_deprecated_helper_emits_warning(self):
        """deprecated_shape_assumption should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sa = deprecated_shape_assumption("M", (64,))
            
            assert len(w) >= 1
            assert issubclass(w[-1].category, DeprecationWarning)
            # Should include migration guide reference
            msg = str(w[-1].message)
            assert "v0.10" in msg or "v0.11" in msg


class TestConversionFunction:
    """Test shape_assumption_to_guard_condition conversion utility."""
    
    def test_conversion_preserves_symbol(self):
        """Converted guard should have same symbol as original."""
        sa = ShapeAssumption(symbol="test_symbol", dims=(128,))
        gc = shape_assumption_to_guard_condition(sa)
        assert gc.symbol == "test_symbol"
    
    def test_conversion_preserves_dims(self):
        """Converted guard should have same dims as original."""
        sa = ShapeAssumption(symbol="N", dims=(32, 64))
        gc = shape_assumption_to_guard_condition(sa)
        assert gc.dims == (32, 64)
    
    def test_conversion_preserves_severity(self):
        """Converted guard should preserve original severity."""
        sa = ShapeAssumption(symbol="M", dims=(128,), severity=GUARD_SEVERITY_SOFT)
        gc = shape_assumption_to_guard_condition(sa)
        assert gc.severity == GUARD_SEVERITY_SOFT
    
    def test_conversion_accepts_override_severity(self):
        """Converter can override severity if provided."""
        sa = ShapeAssumption(symbol="N", dims=(128,), severity=GUARD_SEVERITY_HARD)
        gc = shape_assumption_to_guard_condition(sa, severity=GUARD_SEVERITY_SOFT)
        assert gc.severity == GUARD_SEVERITY_SOFT
    
    def test_conversion_rejects_non_shape_assumption(self):
        """Converter should reject non-ShapeAssumption types."""
        with pytest.raises(TypeError):
            shape_assumption_to_guard_condition("not a ShapeAssumption")
        
        with pytest.raises(TypeError):
            shape_assumption_to_guard_condition(123)


class TestBackwardCompatibility:
    """Verify backward compatibility with v0.9 code."""
    
    def test_shape_assumption_frozen(self):
        """ShapeAssumption should remain frozen (immutable)."""
        sa = ShapeAssumption(symbol="N", dims=(128,))
        with pytest.raises(AttributeError):
            sa.symbol = "new_value"
    
    def test_shape_assumption_hashable(self):
        """ShapeAssumption should be hashable (frozen dataclass)."""
        sa = ShapeAssumption(symbol="N", dims=(128,))
        hash(sa)  # Should not raise
    
    def test_multiple_shape_assumptions_equal(self):
        """Equal ShapeAssumptions should compare equal."""
        sa1 = ShapeAssumption(symbol="N", dims=(128,))
        sa2 = ShapeAssumption(symbol="N", dims=(128,))
        assert sa1 == sa2
    
    def test_shape_assumption_tuple_usage(self):
        """ShapeAssumptions should work in tuples (as in Score)."""
        guards = (
            ShapeAssumption(symbol="N", dims=(128,)),
            ShapeAssumption(symbol="M", dims=(64,)),
        )
        assert len(guards) == 2
        assert all(isinstance(g, ShapeAssumption) for g in guards)
        assert all(isinstance(g, GuardCondition) for g in guards)


class TestIntegrationWithCache:
    """Verify guard conditions work with ScoreCache."""
    
    def test_cache_with_shape_assumptions(self):
        """ScoreCache should handle scores with ShapeAssumptions."""
        from sonata.cache import ScoreCache
        from sonata.score import Score, Task, Dependency, RuntimeTarget
        
        score = Score(
            name="test",
            runtime_target=RuntimeTarget(),
            tasks=(Task(task_id=0, func_id=0, core_type="aic"),),
            dependencies=(Dependency(producer=0, consumer=0),),
            shape_assumptions=(
                ShapeAssumption(symbol="N", dims=(128,), severity=GUARD_SEVERITY_HARD),
            ),
        )
        
        cache = ScoreCache()
        fp = cache.store(score)
        payload = cache.lookup(fp)
        
        assert payload is not None
        assert payload["name"] == "test"
    
    def test_cache_guard_status_all_satisfied(self):
        """Cache lookup should succeed when guard status is ALL_SATISFIED."""
        from sonata.cache import ScoreCache, GuardStatus
        from sonata.score import Score, Task, Dependency, RuntimeTarget
        
        score = Score(
            name="test",
            runtime_target=RuntimeTarget(),
            tasks=(Task(task_id=0, func_id=0, core_type="aic"),),
            dependencies=(),
            shape_assumptions=(),
        )
        
        cache = ScoreCache()
        fp = cache.store(score)
        
        # Default guard status is ALL_SATISFIED
        assert cache.contains(fp)
        assert cache.lookup(fp) is not None
    
    def test_cache_guard_status_failed(self):
        """Cache lookup should fail when guard status is NOT ALL_SATISFIED."""
        from sonata.cache import ScoreCache, GuardStatus, CacheEntry
        from sonata.score import Score, Task, Dependency, RuntimeTarget
        
        score = Score(
            name="test",
            runtime_target=RuntimeTarget(),
            tasks=(Task(task_id=0, func_id=0, core_type="aic"),),
            dependencies=(),
            shape_assumptions=(),
        )
        
        cache = ScoreCache()
        fp = cache.store(score)
        
        # Manually set guard status to FAILED
        existing = cache._entries[fp]
        cache._entries[fp] = CacheEntry(
            fingerprint=existing.fingerprint,
            score_payload=existing.score_payload,
            schema_version=existing.schema_version,
            fingerprint_version=existing.fingerprint_version,
            created_at=existing.created_at,
            plan_handle_payload=existing.plan_handle_payload,
            metadata=existing.metadata,
            guard_status=GuardStatus.ALL_FAILED,
        )
        
        # Should treat as cache miss
        assert not cache.contains(fp)
        assert cache.lookup(fp) is None


class TestCustomGuardSubclasses:
    """Test creating custom GuardCondition subclasses."""
    
    def test_basic_custom_guard(self):
        """Can create basic GuardCondition subclass."""
        class SimpleGuard(GuardCondition):
            def __init__(self, symbol: str):
                super().__init__(symbol=symbol, severity=GUARD_SEVERITY_HARD)
            
            def evaluate(self, runtime_values: dict) -> bool:
                return True
        
        guard = SimpleGuard("test")
        assert isinstance(guard, GuardCondition)
        assert guard.evaluate({}) is True
    
    def test_custom_guard_with_parameters(self):
        """Custom guards can store additional parameters."""
        class ThresholdGuard(GuardCondition):
            def __init__(self, symbol: str, threshold: int):
                super().__init__(symbol=symbol, severity=GUARD_SEVERITY_SOFT)
                self.threshold = threshold
            
            def evaluate(self, runtime_values: dict) -> bool:
                return runtime_values.get(self.symbol, 0) <= self.threshold
        
        guard = ThresholdGuard("temp", 80)
        assert guard.evaluate({"temp": 70}) is True
        assert guard.evaluate({"temp": 90}) is False
    
    def test_custom_guard_severity_property(self):
        """Custom guards respect severity.requires_replan property."""
        class HardGuard(GuardCondition):
            def __init__(self):
                super().__init__("hard", GUARD_SEVERITY_HARD)
            
            def evaluate(self, ctx: dict) -> bool:
                return True
        
        class SoftGuard(GuardCondition):
            def __init__(self):
                super().__init__("soft", GUARD_SEVERITY_SOFT)
            
            def evaluate(self, ctx: dict) -> bool:
                return True
        
        hard = HardGuard()
        soft = SoftGuard()
        
        assert hard.severity.requires_replan is True
        assert soft.severity.requires_replan is False


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_dims(self):
        """ShapeAssumption with empty dims should work."""
        sa = ShapeAssumption(symbol="scalar", dims=())
        assert sa.dims == ()
        assert sa.symbol == "scalar"
    
    def test_large_dims(self):
        """ShapeAssumption with large dimension values should work."""
        sa = ShapeAssumption(symbol="big", dims=(10**6, 10**6))
        assert sa.dims == (10**6, 10**6)
    
    def test_special_symbols(self):
        """ShapeAssumption with special symbol names should work."""
        symbols = ["batch_size", "seq-len", "tensor_0", "123numeric"]
        for sym in symbols:
            sa = ShapeAssumption(symbol=sym, dims=(32,))
            assert sa.symbol == sym
    
    def test_mixed_severity_list(self):
        """List of ShapeAssumptions with mixed severities should work."""
        guards = (
            ShapeAssumption(symbol="a", dims=(1,), severity=GUARD_SEVERITY_HARD),
            ShapeAssumption(symbol="b", dims=(2,), severity=GUARD_SEVERITY_SOFT),
            ShapeAssumption(symbol="c", dims=(3,)),  # Defaults to HARD
        )
        assert len(guards) == 3
        assert guards[0].severity == GUARD_SEVERITY_HARD
        assert guards[1].severity == GUARD_SEVERITY_SOFT
        assert guards[2].severity == GUARD_SEVERITY_HARD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
