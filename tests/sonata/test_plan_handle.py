"""Tests for PlanHandle, FuncRegistry, and RuntimeArgBinding."""

import json
import pytest

from sonata.plan_handle import (
    FuncRegistry,
    FuncRegistryEntry,
    PlanHandle,
    RUNTIME_CONTRACT_VERSION,
    RuntimeArgBinding,
    GuardStatus,
)
from sonata.score import (
    Dependency,
    RuntimeTarget,
    Score,
    ShapeAssumption,
    Task,
    raw_runtime_target,
)
from sonata.guard import GUARD_SEVERITY_HARD, GUARD_SEVERITY_SOFT
from sonata.serialization import (
    plan_handle_to_dict,
    plan_handle_to_json,
    score_fingerprint,
)


def _make_score(**overrides):
    defaults = dict(
        name="test_score",
        runtime_target=RuntimeTarget(
            runtime="host_build_graph", function_name="build_test_graph"
        ),
        tasks=(
            Task(task_id=0, func_id=0, core_type="aic",
                 args=("a", "b"), arg_directions=("input", "output"),
                 arg_storage_keys=("param:a", "alloc:b"), name="matmul"),
            Task(task_id=1, func_id=1, core_type="aiv",
                 args=("b", "c"), arg_directions=("input", "output"),
                 arg_storage_keys=("alloc:b", "alloc:c"), name="add"),
        ),
        dependencies=(Dependency(producer=0, consumer=1),),
        shape_assumptions=(ShapeAssumption(symbol="x", dims=(32, 64), severity="hard"),),
    )
    defaults.update(overrides)
    return Score(**defaults)


class TestFuncRegistry:
    def test_from_score(self):
        score = _make_score()
        registry = FuncRegistry.from_score(score)
        assert registry.names() == frozenset({"matmul", "add"})
        assert registry.by_name("matmul").sonata_func_id == 0
        assert registry.by_name("add").sonata_func_id == 1

    def test_from_score_deduplicates_names(self):
        tasks = (
            Task(task_id=0, func_id=0, core_type="aic", name="matmul"),
            Task(task_id=1, func_id=0, core_type="aic", name="matmul"),
        )
        score = _make_score(tasks=tasks)
        registry = FuncRegistry.from_score(score)
        assert len(registry.entries) == 1

    def test_by_name_not_found(self):
        registry = FuncRegistry()
        assert registry.by_name("nonexistent") is None

    def test_has_unbound_entries(self):
        registry = FuncRegistry(entries=(
            FuncRegistryEntry(name="f", sonata_func_id=0),
        ))
        assert registry.has_unbound_entries()

    def test_bind_runtime_ids(self):
        registry = FuncRegistry(entries=(
            FuncRegistryEntry(name="matmul", sonata_func_id=0),
            FuncRegistryEntry(name="add", sonata_func_id=1),
        ))
        bound = registry.bind_runtime_ids({"matmul": 10, "add": 20})
        assert bound.by_name("matmul").runtime_func_id == 10
        assert bound.by_name("add").runtime_func_id == 20

    def test_bind_runtime_ids_partial(self):
        registry = FuncRegistry(entries=(
            FuncRegistryEntry(name="matmul", sonata_func_id=0),
            FuncRegistryEntry(name="add", sonata_func_id=1),
        ))
        bound = registry.bind_runtime_ids({"matmul": 10})
        assert bound.by_name("matmul").runtime_func_id == 10
        assert bound.by_name("add").runtime_func_id is None

    def test_sonata_func_ids(self):
        registry = FuncRegistry(entries=(
            FuncRegistryEntry(name="a", sonata_func_id=0),
            FuncRegistryEntry(name="b", sonata_func_id=1),
        ))
        assert registry.sonata_func_ids() == {"a": 0, "b": 1}

    def test_empty_registry(self):
        registry = FuncRegistry()
        assert registry.names() == frozenset()
        assert not registry.has_unbound_entries()


class TestPlanHandle:
    def test_from_score(self):
        score = _make_score()
        ph = PlanHandle.from_score(score)
        assert ph.score_fingerprint == score_fingerprint(score)
        assert ph.runtime_target == raw_runtime_target(score)
        assert ph.source_adapter == "post_simplify"
        assert ph.runtime_contract_version == RUNTIME_CONTRACT_VERSION
        assert ph.func_registry.names() == frozenset({"matmul", "add"})

    def test_from_score_custom_adapter(self):
        score = _make_score()
        ph = PlanHandle.from_score(score, source_adapter="post_materialize")
        assert ph.source_adapter == "post_materialize"

    def test_from_score_custom_runtime_target(self):
        score = _make_score()
        target = RuntimeTarget(runtime="custom", function_name="custom_fn")
        ph = PlanHandle.from_score(score, runtime_target=target)
        assert ph.runtime_target == target

    def test_plan_handle_frozen(self):
        score = _make_score()
        ph = PlanHandle.from_score(score)
        with pytest.raises(AttributeError):
            ph.score_fingerprint = "changed"


class TestRuntimeArgBinding:
    def test_binding_creation(self):
        binding = RuntimeArgBinding(
            task_id=0, arg_index=0, storage_key="param:a",
            direction="input", runtime_handle=42,
        )
        assert binding.task_id == 0
        assert binding.runtime_handle == 42

    def test_binding_frozen(self):
        binding = RuntimeArgBinding(
            task_id=0, arg_index=0, storage_key=None,
            direction="input", runtime_handle=0,
        )
        with pytest.raises(AttributeError):
            binding.task_id = 1


class TestPlanHandleSerialization:
    def test_plan_handle_to_dict(self):
        score = _make_score()
        ph = PlanHandle.from_score(score)
        d = plan_handle_to_dict(ph)
        assert d["schema_version"] == 1
        assert d["score_fingerprint"] == score_fingerprint(score)
        assert d["source_adapter"] == "post_simplify"
        assert d["runtime_contract_version"] == RUNTIME_CONTRACT_VERSION
        assert len(d["func_registry"]) == 2

    def test_plan_handle_to_json(self):
        score = _make_score()
        ph = PlanHandle.from_score(score)
        text = plan_handle_to_json(ph)
        parsed = json.loads(text)
        assert parsed["score_fingerprint"] == score_fingerprint(score)

    def test_plan_handle_dict_deterministic(self):
        score = _make_score()
        ph = PlanHandle.from_score(score)
        assert plan_handle_to_dict(ph) == plan_handle_to_dict(ph)

    def test_plan_handle_with_bindings(self):
        score = _make_score()
        bindings = (
            RuntimeArgBinding(task_id=0, arg_index=0, storage_key="param:a",
                              direction="input", runtime_handle=0),
        )
        ph = PlanHandle.from_score(score)
        ph = PlanHandle(
            score_fingerprint=ph.score_fingerprint,
            runtime_target=ph.runtime_target,
            source_adapter=ph.source_adapter,
            func_registry=ph.func_registry,
            arg_bindings=bindings,
        )
        d = plan_handle_to_dict(ph)
        assert len(d["arg_bindings"]) == 1
        assert d["arg_bindings"][0]["storage_key"] == "param:a"


class TestEndToEndSmoke:
    """Score -> PlanHandle -> HostBuildGraphPlan full pipeline smoke."""

    def test_single_task_smoke(self):
        score = Score(
            name="single_op",
            runtime_target=RuntimeTarget(
                runtime="host_build_graph",
                function_name="build_single_op_graph",
            ),
            tasks=(
                Task(task_id=0, func_id=0, core_type="aic",
                     args=("x", "y"), arg_directions=("input", "output"),
                     arg_storage_keys=("param:x", "alloc:y"), name="matmul"),
            ),
            dependencies=(),
            shape_assumptions=(ShapeAssumption(symbol="x", dims=(64, 64), severity="hard"),),
        )
        ph = PlanHandle.from_score(score)
        assert ph.score_fingerprint == score_fingerprint(score)
        assert ph.func_registry.names() == frozenset({"matmul"})

        from sonata.runtime_adapter import HostBuildGraphRuntimeAdapter
        bindings = (
            RuntimeArgBinding(task_id=0, arg_index=0, storage_key="param:x",
                              direction="input", runtime_handle=0),
            RuntimeArgBinding(task_id=0, arg_index=1, storage_key="alloc:y",
                              direction="output", runtime_handle=1),
        )
        ph = PlanHandle(
            score_fingerprint=ph.score_fingerprint,
            runtime_target=ph.runtime_target,
            source_adapter=ph.source_adapter,
            func_registry=ph.func_registry,
            arg_bindings=bindings,
        )
        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(score, ph)
        assert result.success
        assert result.plan.task_count() == 1
        assert result.plan.tasks[0].name == "matmul"

    def test_multi_task_smoke(self):
        score = _make_score()
        ph = PlanHandle.from_score(score)

        from sonata.directions import MEMORY_DIRECTIONS, normalize_direction
        bindings = []
        for task in score.tasks:
            for i, d in enumerate(task.arg_directions):
                if normalize_direction(d) in MEMORY_DIRECTIONS:
                    sk = task.arg_storage_keys[i] if task.arg_storage_keys else None
                    bindings.append(RuntimeArgBinding(
                        task_id=task.task_id, arg_index=i,
                        storage_key=sk, direction=d, runtime_handle=i,
                    ))
        ph = PlanHandle(
            score_fingerprint=ph.score_fingerprint,
            runtime_target=ph.runtime_target,
            source_adapter=ph.source_adapter,
            func_registry=ph.func_registry,
            arg_bindings=tuple(bindings),
        )

        from sonata.runtime_adapter import HostBuildGraphRuntimeAdapter
        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(score, ph)
        assert result.success
        assert result.plan.task_count() == 2
        assert result.plan.edge_count() == 1
        assert result.plan.metadata["source_adapter"] == "post_simplify"

    def test_guard_invalidation_smoke(self):
        """Shape change -> different fingerprint -> adapter rejects."""
        score_a = Score(
            name="guard_test",
            runtime_target=RuntimeTarget(runtime="host_build_graph",
                                         function_name="build_guard_graph"),
            tasks=(
                Task(task_id=0, func_id=0, core_type="aic",
                     args=("x",), arg_directions=("input",),
                     arg_storage_keys=("param:x",), name="op"),
            ),
            shape_assumptions=(ShapeAssumption(symbol="x", dims=(32,), severity="hard"),),
        )
        score_b = Score(
            name="guard_test",
            runtime_target=RuntimeTarget(runtime="host_build_graph",
                                         function_name="build_guard_graph"),
            tasks=(
                Task(task_id=0, func_id=0, core_type="aic",
                     args=("x",), arg_directions=("input",),
                     arg_storage_keys=("param:x",), name="op"),
            ),
            shape_assumptions=(ShapeAssumption(symbol="x", dims=(64,), severity="hard"),),
        )
        fp_a = score_fingerprint(score_a)
        fp_b = score_fingerprint(score_b)
        assert fp_a != fp_b

        from sonata.runtime_adapter import HostBuildGraphRuntimeAdapter
        ph_a = PlanHandle.from_score(score_a)
        bindings_a = (
            RuntimeArgBinding(task_id=0, arg_index=0, storage_key="param:x",
                              direction="input", runtime_handle=0),
        )
        ph_a = PlanHandle(
            score_fingerprint=ph_a.score_fingerprint,
            runtime_target=ph_a.runtime_target,
            source_adapter=ph_a.source_adapter,
            func_registry=ph_a.func_registry,
            arg_bindings=bindings_a,
        )
        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(score_b, ph_a)
        assert not result.success
        codes = [r.code for r in result.reasons]
        assert "runtime_adapter_fingerprint_mismatch" in codes


# ============================================================================
# Phase 4: PlanHandle Guard Status Integration Tests
# ============================================================================

class TestPlanHandleGuardStatus:
    """Tests for guard status integration in PlanHandle (v0.10+)."""
    
    def test_guard_status_default_all_satisfied(self):
        """PlanHandle should default to ALL_SATISFIED guard status."""
        score = Score(
            name="test",
            runtime_target=RuntimeTarget(runtime="host_build_graph",
                                         function_name="build_test"),
            tasks=(
                Task(task_id=0, func_id=0, core_type="aic",
                     args=("x",), arg_directions=("input",),
                     arg_storage_keys=("param:x",), name="op"),
            ),
        )
        ph = PlanHandle.from_score(score)
        assert ph.guard_status == GuardStatus.ALL_SATISFIED
    
    def test_guard_status_custom_value(self):
        """PlanHandle should accept custom guard_status."""
        score = Score(
            name="test",
            runtime_target=RuntimeTarget(runtime="host_build_graph",
                                         function_name="build_test"),
            tasks=(
                Task(task_id=0, func_id=0, core_type="aic",
                     args=("x",), arg_directions=("input",),
                     arg_storage_keys=("param:x",), name="op"),
            ),
        )
        ph = PlanHandle.from_score(score)
        # Create new PlanHandle with different guard status
        ph_modified = PlanHandle(
            score_fingerprint=ph.score_fingerprint,
            runtime_target=ph.runtime_target,
            source_adapter=ph.source_adapter,
            func_registry=ph.func_registry,
            guard_status=GuardStatus.PARTIAL_FAILED,
        )
        assert ph_modified.guard_status == GuardStatus.PARTIAL_FAILED
    
    def test_critical_guards_empty_by_default(self):
        """critical_guards should be empty tuple by default."""
        score = Score(
            name="test",
            runtime_target=RuntimeTarget(runtime="host_build_graph",
                                         function_name="build_test"),
            tasks=(
                Task(task_id=0, func_id=0, core_type="aic",
                     args=("x",), arg_directions=("input",),
                     arg_storage_keys=("param:x",), name="op"),
            ),
        )
        ph = PlanHandle.from_score(score)
        assert ph.critical_guards == ()
    
    def test_critical_guards_with_values(self):
        """critical_guards should accept GuardCondition instances."""
        from sonata.guard import ShapeAssumption
        
        score = Score(
            name="test",
            runtime_target=RuntimeTarget(runtime="host_build_graph",
                                         function_name="build_test"),
            tasks=(
                Task(task_id=0, func_id=0, core_type="aic",
                     args=("x",), arg_directions=("input",),
                     arg_storage_keys=("param:x",), name="op"),
            ),
        )
        ph = PlanHandle.from_score(score)
        
        # Add critical guards
        critical = (
            ShapeAssumption(symbol="batch_size", dims=(32,), severity=GUARD_SEVERITY_HARD),
            ShapeAssumption(symbol="seq_len", dims=(128,), severity=GUARD_SEVERITY_SOFT),
        )
        ph_modified = PlanHandle(
            score_fingerprint=ph.score_fingerprint,
            runtime_target=ph.runtime_target,
            source_adapter=ph.source_adapter,
            func_registry=ph.func_registry,
            critical_guards=critical,
        )
        assert len(ph_modified.critical_guards) == 2
        assert ph_modified.critical_guards[0].symbol == "batch_size"
    
    def test_guard_status_in_metadata(self):
        """guard_status should be included in metadata dict."""
        score = Score(
            name="test",
            runtime_target=RuntimeTarget(runtime="host_build_graph",
                                         function_name="build_test"),
            tasks=(
                Task(task_id=0, func_id=0, core_type="aic",
                     args=("x",), arg_directions=("input",),
                     arg_storage_keys=("param:x",), name="op"),
            ),
        )
        ph = PlanHandle.from_score(score)
        
        d = plan_handle_to_dict(ph)
        assert "guard_status" in d
        assert d["guard_status"] == "all_satisfied"
    
    def test_guard_evaluator_integration(self):
        """Integration test: use GuardEvaluator to update PlanHandle status."""
        from sonata.guard import GuardEvaluator
        from sonata.plan_handle import GuardStatus
        
        score = Score(
            name="test",
            runtime_target=RuntimeTarget(runtime="host_build_graph",
                                         function_name="build_test"),
            tasks=(
                Task(task_id=0, func_id=0, core_type="aic",
                     args=("x",), arg_directions=("input",),
                     arg_storage_keys=("param:x",), name="op"),
            ),
            shape_assumptions=(
                ShapeAssumption(symbol="batch_size", dims=(32,), severity=GUARD_SEVERITY_HARD),
                ShapeAssumption(symbol="seq_len", dims=(128,), severity=GUARD_SEVERITY_SOFT),
            ),
        )
        ph = PlanHandle.from_score(score)
        
        evaluator = GuardEvaluator()
        
        # All satisfied case
        runtime_values = {"batch_size": [32], "seq_len": [128]}
        all_satisfied, results = evaluator.evaluate_all(
            score.shape_assumptions, runtime_values
        )
        
        if all_satisfied:
            ph = PlanHandle(
                score_fingerprint=ph.score_fingerprint,
                runtime_target=ph.runtime_target,
                source_adapter=ph.source_adapter,
                func_registry=ph.func_registry,
                guard_status=GuardStatus.ALL_SATISFIED,
                critical_guards=score.shape_assumptions,
            )
            assert ph.guard_status == GuardStatus.ALL_SATISFIED
        
        # Partial failure case
        runtime_values = {"batch_size": [32], "seq_len": [256]}  # seq_len changed
        all_satisfied, results = evaluator.evaluate_all(
            score.shape_assumptions, runtime_values
        )
        
        if not all_satisfied:
            # Check if any hard guard failed
            has_hard_failure = any(
                not satisfied and guard.severity.requires_replan
                for guard, satisfied, _ in results
            )
            
            if has_hard_failure:
                new_status = GuardStatus.ALL_FAILED
            else:
                new_status = GuardStatus.PARTIAL_FAILED
            
            ph = PlanHandle(
                score_fingerprint=ph.score_fingerprint,
                runtime_target=ph.runtime_target,
                source_adapter=ph.source_adapter,
                func_registry=ph.func_registry,
                guard_status=new_status,
                critical_guards=score.shape_assumptions,
            )
            assert ph.guard_status == new_status

    def test_stale_status_value(self):
        """GuardStatus.STALE is a valid enum value."""
        assert GuardStatus.STALE.value == "stale"

    def test_stale_status_on_plan_handle(self):
        """PlanHandle can be set to STALE status."""
        score = Score(
            name="test",
            runtime_target=RuntimeTarget(runtime="host_build_graph",
                                         function_name="build_test"),
            tasks=(
                Task(task_id=0, func_id=0, core_type="aic",
                     args=("x",), arg_directions=("input",),
                     arg_storage_keys=("param:x",), name="op"),
            ),
        )
        ph = PlanHandle.from_score(score)
        ph_stale = PlanHandle(
            score_fingerprint=ph.score_fingerprint,
            runtime_target=ph.runtime_target,
            source_adapter=ph.source_adapter,
            func_registry=ph.func_registry,
            guard_status=GuardStatus.STALE,
        )
        assert ph_stale.guard_status == GuardStatus.STALE

    def test_stale_serialization(self):
        """STALE guard status serializes correctly."""
        score = Score(
            name="test",
            runtime_target=RuntimeTarget(runtime="host_build_graph",
                                         function_name="build_test"),
            tasks=(
                Task(task_id=0, func_id=0, core_type="aic",
                     args=("x",), arg_directions=("input",),
                     arg_storage_keys=("param:x",), name="op"),
            ),
        )
        ph = PlanHandle.from_score(score)
        ph_stale = PlanHandle(
            score_fingerprint=ph.score_fingerprint,
            runtime_target=ph.runtime_target,
            source_adapter=ph.source_adapter,
            func_registry=ph.func_registry,
            guard_status=GuardStatus.STALE,
        )
        d = plan_handle_to_dict(ph_stale)
        assert d["guard_status"] == "stale"

    def test_stale_vs_all_failed_semantics(self):
        """STALE means plan handle invalid but Score still valid.
        ALL_FAILED means both plan handle and Score are invalid."""
        assert GuardStatus.STALE != GuardStatus.ALL_FAILED
        assert GuardStatus.STALE != GuardStatus.ALL_SATISFIED
        # STALE should be usable for shape-change scenarios
        # where Score fingerprint doesn't change
        assert GuardStatus.STALE.value == "stale"
