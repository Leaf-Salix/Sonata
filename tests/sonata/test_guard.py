# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for guard condition abstraction (v0.10+).

Covers:
- Phase 1: GuardCondition ABC + ShapeAssumption subclass + GuardSeverity enum
- Phase 2: GuardEvaluator + GuardInvalidator strategies
- Phase 3: Over-guarding mitigation (GuardSelector, density check)
"""

import pytest
from sonata.guard import (
    GuardCondition,
    GuardSeverity,
    GUARD_SEVERITY_SOFT,
    GUARD_SEVERITY_HARD,
    ShapeAssumption,
    InvalidateAction,
    GuardEvaluator,
    GuardInvalidator,
    GuardSelector,
    EntryParamGuardSelector,
    check_guard_density,
)


# ============================================================================
# Phase 1: GuardCondition ABC Tests
# ============================================================================

class TestGuardSeverity:
    """Tests for GuardSeverity enum."""
    
    def test_soft_requires_no_replan(self):
        assert not GUARD_SEVERITY_SOFT.requires_replan
    
    def test_hard_requires_replan(self):
        assert GUARD_SEVERITY_HARD.requires_replan
    
    def test_str_representation(self):
        assert str(GUARD_SEVERITY_SOFT) == "soft"
        assert str(GUARD_SEVERITY_HARD) == "hard"
    
    def test_repr_format(self):
        assert repr(GUARD_SEVERITY_SOFT) == "GuardSeverity('soft')"
        assert repr(GUARD_SEVERITY_HARD) == "GuardSeverity('hard')"


class TestShapeAssumption:
    """Tests for ShapeAssumption as GuardCondition subclass."""
    
    def test_construction_default_severity(self):
        guard = ShapeAssumption(symbol="batch_size", dims=(32,))
        assert guard.severity == GUARD_SEVERITY_HARD
    
    def test_construction_custom_severity(self):
        guard = ShapeAssumption(
            symbol="seq_len", 
            dims=(128,), 
            severity=GUARD_SEVERITY_SOFT
        )
        assert guard.severity == GUARD_SEVERITY_SOFT
    
    def test_evaluate_match(self):
        guard = ShapeAssumption(symbol="batch_size", dims=(32, 64))
        assert guard.evaluate({"batch_size": [32, 64]}) is True
    
    def test_evaluate_mismatch(self):
        guard = ShapeAssumption(symbol="batch_size", dims=(32,))
        assert guard.evaluate({"batch_size": [64]}) is False
    
    def test_evaluate_missing_key(self):
        guard = ShapeAssumption(symbol="batch_size", dims=(32,))
        assert guard.evaluate({}) is False
    
    def test_evaluate_scalar(self):
        guard = ShapeAssumption(symbol="scalar_val", dims=())
        assert guard.evaluate({"scalar_val": ()}) is True
        assert guard.evaluate({"scalar_val": [1]}) is False
    
    def test_equality(self):
        g1 = ShapeAssumption(symbol="x", dims=(32,))
        g2 = ShapeAssumption(symbol="x", dims=(32,))
        g3 = ShapeAssumption(symbol="y", dims=(32,))
        assert g1 == g2
        assert g1 != g3
    
    def test_hash_consistency(self):
        g1 = ShapeAssumption(symbol="x", dims=(32,))
        g2 = ShapeAssumption(symbol="x", dims=(32,))
        assert hash(g1) == hash(g2)
    
    def test_to_dict_with_dims(self):
        guard = ShapeAssumption(symbol="batch_size", dims=(32, 64))
        d = guard.to_dict()
        assert d["type"] == "ShapeAssumption"
        assert d["symbol"] == "batch_size"
        assert d["dims"] == (32, 64)
        assert d["severity"] == "hard"
    
    def test_from_dict_with_type_field(self):
        data = {
            "type": "ShapeAssumption",
            "symbol": "batch_size",
            "dims": [32, 64],
            "severity": "soft"
        }
        guard = ShapeAssumption.from_dict(data)
        assert isinstance(guard, ShapeAssumption)
        assert guard.symbol == "batch_size"
        assert guard.dims == (32, 64)
        assert guard.severity == GUARD_SEVERITY_SOFT
    
    def test_from_dict_backward_compatible(self):
        # Old format without type field
        data = {
            "symbol": "batch_size",
            "dims": [32],
            "severity": "hard"
        }
        guard = ShapeAssumption.from_dict(data)
        assert isinstance(guard, ShapeAssumption)
        assert guard.dims == (32,)


# ============================================================================
# Phase 2: GuardEvaluator & GuardInvalidator Tests
# ============================================================================

class TestGuardEvaluator:
    """Tests for GuardEvaluator."""
    
    def test_evaluate_satisfied_hard_guard(self):
        evaluator = GuardEvaluator()
        guard = ShapeAssumption(symbol="batch_size", dims=(32,))
        satisfied, action = evaluator.evaluate(guard, {"batch_size": [32]})
        
        assert satisfied is True
        assert action == InvalidateAction.REPLAN  # hard guard → REPLAN if violated
    
    def test_evaluate_violated_hard_guard(self):
        evaluator = GuardEvaluator()
        guard = ShapeAssumption(symbol="batch_size", dims=(32,))
        satisfied, action = evaluator.evaluate(guard, {"batch_size": [64]})
        
        assert satisfied is False
        assert action == InvalidateAction.REPLAN
    
    def test_evaluate_satisfied_soft_guard(self):
        evaluator = GuardEvaluator()
        guard = ShapeAssumption(
            symbol="temp_limit", 
            dims=(), 
            severity=GUARD_SEVERITY_SOFT
        )
        satisfied, action = evaluator.evaluate(guard, {"temp_limit": []})
        
        assert satisfied is True
        assert action == InvalidateAction.INVALIDATE_HANDLE  # soft guard
    
    def test_evaluate_all_satisfied(self):
        evaluator = GuardEvaluator()
        guards = [
            ShapeAssumption(symbol="a", dims=(1,)),
            ShapeAssumption(symbol="b", dims=(2,)),
        ]
        all_satisfied, results = evaluator.evaluate_all(
            guards, {"a": [1], "b": [2]}
        )
        
        assert all_satisfied is True
        assert len(results) == 2
        for guard, satisfied, _ in results:
            assert satisfied is True
    
    def test_evaluate_all_partial_failure(self):
        evaluator = GuardEvaluator()
        guards = [
            ShapeAssumption(symbol="a", dims=(1,)),
            ShapeAssumption(symbol="b", dims=(2,)),
        ]
        all_satisfied, results = evaluator.evaluate_all(
            guards, {"a": [1], "b": [3]}
        )
        
        assert all_satisfied is False
        assert len(results) == 2
        
        # Check which one failed
        failures = [g for g, s, _ in results if not s]
        assert len(failures) == 1
        assert failures[0].symbol == "b"
    
    def test_evaluate_error_treats_as_violation(self):
        """On evaluation error, treat as violation with conservative action."""
        evaluator = GuardEvaluator()
        
        class BrokenGuard(GuardCondition):
            def evaluate(self, runtime_values):
                raise RuntimeError("Intentional error")
        
        guard = BrokenGuard(symbol="broken", severity=GUARD_SEVERITY_HARD)
        satisfied, action = evaluator.evaluate(guard, {})
        
        assert satisfied is False
        assert action == InvalidateAction.REPLAN  # Conservative on error


class TestGuardInvalidator:
    """Tests for GuardInvalidator strategies."""
    
    def test_invalidate_hard_guard(self):
        invalidator = GuardInvalidator()
        guard = ShapeAssumption(symbol="batch_size", dims=(32,))
        action = invalidator.invalidate(guard, reason="shape mismatch")
        
        assert action == InvalidateAction.REPLAN
    
    def test_invalidate_soft_guard(self):
        invalidator = GuardInvalidator()
        guard = ShapeAssumption(
            symbol="temp", 
            dims=(), 
            severity=GUARD_SEVERITY_SOFT
        )
        action = invalidator.invalidate(guard, reason="threshold exceeded")
        
        assert action == InvalidateAction.INVALIDATE_HANDLE
    
    def test_invalidate_all_with_hard_violation(self):
        invalidator = GuardInvalidator()
        results = [
            (ShapeAssumption("a", dims=(1,)), False, InvalidateAction.REPLAN),
            (ShapeAssumption("b", dims=(2,)), True, InvalidateAction.REPLAN),
        ]
        action = invalidator.invalidate_all(results)
        
        assert action == InvalidateAction.REPLAN  # Any hard → REPLAN
    
    def test_invalidate_all_only_soft_violations(self):
        invalidator = GuardInvalidator()
        results = [
            (ShapeAssumption("a", dims=(1,), severity=GUARD_SEVERITY_SOFT), False, InvalidateAction.INVALIDATE_HANDLE),
            (ShapeAssumption("b", dims=(2,), severity=GUARD_SEVERITY_SOFT), False, InvalidateAction.INVALIDATE_HANDLE),
        ]
        action = invalidator.invalidate_all(results)
        
        assert action == InvalidateAction.INVALIDATE_HANDLE


# ============================================================================
# Phase 3: Over-guarding Mitigation Tests
# ============================================================================

class TestGuardSelector:
    """Tests for GuardSelector interface and implementations."""
    
    def test_abstract_base_class(self):
        """GuardSelector should be abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            GuardSelector()  # type: ignore
    
    def test_entry_param_selector_selects_all_when_no_top_k(self):
        selector = EntryParamGuardSelector()
        guards = [
            ShapeAssumption(symbol="input_a", dims=(32,)),
            ShapeAssumption(symbol="input_b", dims=(64,)),
            ShapeAssumption(symbol="intermediate", dims=(16,)),
        ]
        selected = selector.select(guards)
        
        assert len(selected) == 3
        assert selected == guards  # All selected when no top_k limit
    
    def test_entry_param_selector_applies_top_k(self):
        selector = EntryParamGuardSelector(top_k=2)
        guards = [
            ShapeAssumption(symbol="a", dims=(1,)),
            ShapeAssumption(symbol="b", dims=(2,)),
            ShapeAssumption(symbol="c", dims=(3,)),
            ShapeAssumption(symbol="d", dims=(4,)),
        ]
        selected = selector.select(guards)
        
        assert len(selected) == 2
        assert selected == guards[:2]  # First 2 only
    
    def test_entry_param_selector_prioritizes_priority_symbols(self):
        selector = EntryParamGuardSelector(
            top_k=3,
            priority_symbols=["critical_a", "critical_b"]
        )
        guards = [
            ShapeAssumption(symbol="normal_1", dims=(1,)),
            ShapeAssumption(symbol="critical_a", dims=(2,)),
            ShapeAssumption(symbol="normal_2", dims=(3,)),
            ShapeAssumption(symbol="critical_b", dims=(4,)),
            ShapeAssumption(symbol="normal_3", dims=(5,)),
        ]
        selected = selector.select(guards)
        
        # Priority symbols should come first
        selected_symbols = [g.symbol for g in selected]
        assert "critical_a" in selected_symbols[:2]
        assert "critical_b" in selected_symbols[:2]
        assert len(selected) == 3  # top_k limit


class TestCheckGuardDensity:
    """Tests for guard density checking utility."""
    
    def test_below_threshold(self):
        guards = [
            ShapeAssumption(symbol=f"x{i}", dims=(1,))
            for i in range(10)
        ]
        exceeds, count = check_guard_density(guards, warning_threshold=50)
        
        assert exceeds is False
        assert count == 10
    
    def test_above_threshold(self):
        guards = [
            ShapeAssumption(symbol=f"x{i}", dims=(1,))
            for i in range(100)
        ]
        exceeds, count = check_guard_density(guards, warning_threshold=50)
        
        assert exceeds is True
        assert count == 100
    
    def test_at_threshold(self):
        guards = [
            ShapeAssumption(symbol=f"x{i}", dims=(1,))
            for i in range(50)
        ]
        exceeds, count = check_guard_density(guards, warning_threshold=50)
        
        assert exceeds is False  # Not > 50, exactly 50
        assert count == 50
    
    def test_logger_callback_on_exceed(self):
        warnings = []
        def logger_fn(threshold, message):
            warnings.append((threshold, message))
        
        guards = [
            ShapeAssumption(symbol=f"x{i}", dims=(1,))
            for i in range(100)
        ]
        check_guard_density(guards, warning_threshold=50, logger=logger_fn)
        
        assert len(warnings) == 1
        threshold, message = warnings[0]
        assert threshold == 50
        assert "Guard density 100 exceeds threshold 50" in message


# ============================================================================
# Integration Tests
# ============================================================================

class TestGuardIntegration:
    """End-to-end integration tests for guard system."""
    
    def test_full_workflow_evaluator_then_invalidator(self):
        """Test complete workflow: evaluate → decide action → invalidate."""
        evaluator = GuardEvaluator()
        invalidator = GuardInvalidator()
        
        # Create a batch of guards
        guards = [
            ShapeAssumption(symbol="batch_size", dims=(32,), severity=GUARD_SEVERITY_HARD),
            ShapeAssumption(symbol="seq_len", dims=(128,), severity=GUARD_SEVERITY_SOFT),
        ]
        
        # Simulate runtime values where seq_len changed but batch_size didn't
        runtime_values = {"batch_size": [32], "seq_len": [256]}
        
        # Evaluate all guards
        all_satisfied, results = evaluator.evaluate_all(guards, runtime_values)
        
        assert all_satisfied is False  # seq_len violated
        
        # Determine overall action
        action = invalidator.invalidate_all(results)
        
        # Should be REPLAN because batch_size is HARD even though it passed
        # But seq_len is SOFT and failed → INVALIDATE_HANDLE
        # Actually policy: any HARD violation → REPLAN, else INVALIDATE_HANDLE
        # Since batch_size passed, only seq_len failed (SOFT) → INVALIDATE_HANDLE
        assert action == InvalidateAction.INVALIDATE_HANDLE
    
    def test_over_guarding_detection_and_selection(self):
        """Test detecting over-guarding and using selector to mitigate."""
        # Create 100 guards (simulating over-guarding scenario)
        all_guards = [
            ShapeAssumption(symbol=f"param_{i}", dims=(32,))
            for i in range(100)
        ]
        
        # Detect over-guarding
        exceeds, count = check_guard_density(all_guards, warning_threshold=50)
        assert exceeds is True
        assert count == 100
        
        # Use selector to reduce to critical guards
        selector = EntryParamGuardSelector(top_k=10)
        critical_guards = selector.select(all_guards)
        
        assert len(critical_guards) == 10
        assert len(critical_guards) < len(all_guards)


# ============================================================================
# Edge Cases
# ============================================================================

class TestGuardEdgeCases:
    """Edge case tests for guard system."""
    
    def test_empty_dims_tuple(self):
        """Scalar tensors have empty dims tuple."""
        guard = ShapeAssumption(symbol="scalar", dims=())
        assert guard.evaluate({"scalar": ()}) is True
        assert guard.evaluate({"scalar": [1]}) is False
    
    def test_nested_list_conversion(self):
        """Guard should handle both list and tuple runtime shapes."""
        guard = ShapeAssumption(symbol="matrix", dims=(32, 64))
        assert guard.evaluate({"matrix": [32, 64]}) is True
        assert guard.evaluate({"matrix": (32, 64)}) is True
    
    def test_guard_with_none_runtime_value(self):
        """Missing runtime value should return False (violation)."""
        guard = ShapeAssumption(symbol="batch_size", dims=(32,))
        assert guard.evaluate({}) is False
        assert guard.evaluate({"other_key": [1]}) is False
    
    def test_invalidate_action_constants(self):
        """Verify InvalidateAction enum values and methods."""
        assert InvalidateAction.REPLAN.value == "replan"
        assert InvalidateAction.INVALIDATE_HANDLE.value == "invalidate_handle"
        assert InvalidateAction.UPDATE_IN_PLACE.value == "update_in_place"
        
        assert InvalidateAction.REPLAN.requires_full_replan() is True
        assert InvalidateAction.INVALIDATE_HANDLE.requires_full_replan() is False
        
        assert InvalidateAction.REPLAN.is_conservative() is True
        assert InvalidateAction.INVALIDATE_HANDLE.is_conservative() is True
        assert InvalidateAction.UPDATE_IN_PLACE.is_conservative() is False
