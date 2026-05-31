"""Tests for HostBuildGraphRuntimeAdapter."""

import pytest

from sonata.fallback import FallbackCode
from sonata.plan_handle import (
    FuncRegistry,
    FuncRegistryEntry,
    PlanHandle,
    RuntimeArgBinding,
)
from sonata.runtime_adapter import (
    HostBuildGraphPlan,
    HostBuildGraphRuntimeAdapter,
    HostBuildGraphTask,
    RuntimeAdapterResult,
)
from sonata.score import (
    Dependency,
    RuntimeTarget,
    Score,
    ShapeAssumption,
    Task,
)
from sonata.serialization import score_fingerprint


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
        shape_assumptions=(ShapeAssumption(symbol="x", dims=(32, 64)),),
    )
    defaults.update(overrides)
    return Score(**defaults)


def _make_bindings(score):
    bindings = []
    for task in score.tasks:
        for i, (arg, d) in enumerate(zip(task.args, task.arg_directions)):
            from sonata.directions import MEMORY_DIRECTIONS, normalize_direction
            if normalize_direction(d) in MEMORY_DIRECTIONS:
                sk = task.arg_storage_keys[i] if task.arg_storage_keys else None
                bindings.append(RuntimeArgBinding(
                    task_id=task.task_id, arg_index=i,
                    storage_key=sk, direction=d,
                    runtime_handle=i,
                ))
    return tuple(bindings)


def _make_plan_handle(score, **overrides):
    defaults = dict(
        score_fingerprint=score_fingerprint(score),
        runtime_target=score.runtime_target,
        source_adapter="post_simplify",
        func_registry=FuncRegistry.from_score(score),
        arg_bindings=_make_bindings(score),
    )
    defaults.update(overrides)
    return PlanHandle(**defaults)


class TestHostBuildGraphRuntimeAdapter:
    def test_generate_success(self):
        score = _make_score()
        ph = _make_plan_handle(score)
        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(score, ph)
        assert result.success
        assert result.plan is not None
        assert result.plan.task_count() == 2
        assert result.plan.edge_count() == 1

    def test_generate_task_details(self):
        score = _make_score()
        ph = _make_plan_handle(score)
        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(score, ph)
        assert result.plan.tasks[0].task_id == 0
        assert result.plan.tasks[0].core_type == "aic"
        assert result.plan.tasks[0].name == "matmul"
        assert result.plan.tasks[1].task_id == 1
        assert result.plan.tasks[1].core_type == "aiv"

    def test_generate_with_runtime_func_ids(self):
        score = _make_score()
        registry = FuncRegistry.from_score(score).bind_runtime_ids(
            {"matmul": 100, "add": 200}
        )
        ph = _make_plan_handle(score, func_registry=registry)
        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(score, ph)
        assert result.success
        assert result.plan.tasks[0].func_id == 100
        assert result.plan.tasks[1].func_id == 200

    def test_generate_metadata(self):
        score = _make_score()
        ph = _make_plan_handle(score)
        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(score, ph)
        meta = result.plan.metadata
        assert meta["score_fingerprint"] == ph.score_fingerprint
        assert meta["source_adapter"] == "post_simplify"
        assert meta["runtime_target"] == "host_build_graph"

    def test_fingerprint_mismatch(self):
        score = _make_score()
        ph = _make_plan_handle(score, score_fingerprint="wrong_hash")
        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(score, ph)
        assert not result.success
        codes = [r.code for r in result.reasons]
        assert FallbackCode.RUNTIME_ADAPTER_FINGERPRINT_MISMATCH.value in codes

    def test_contract_version_mismatch(self):
        score = _make_score()
        ph = _make_plan_handle(score, runtime_contract_version=999)
        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(score, ph)
        assert not result.success
        codes = [r.code for r in result.reasons]
        assert FallbackCode.RUNTIME_ADAPTER_CONTRACT_VERSION_MISMATCH.value in codes

    def test_func_not_registered(self):
        score = _make_score()
        empty_registry = FuncRegistry()
        ph = _make_plan_handle(score, func_registry=empty_registry)
        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(score, ph)
        assert not result.success
        codes = [r.code for r in result.reasons]
        assert FallbackCode.RUNTIME_ADAPTER_FUNC_NOT_REGISTERED.value in codes

    def test_func_unreferenced(self):
        score = _make_score()
        extra_registry = FuncRegistry(entries=(
            FuncRegistryEntry(name="matmul", sonata_func_id=0),
            FuncRegistryEntry(name="add", sonata_func_id=1),
            FuncRegistryEntry(name="ghost_func", sonata_func_id=99),
        ))
        ph = _make_plan_handle(score, func_registry=extra_registry)
        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(score, ph)
        assert not result.success
        codes = [r.code for r in result.reasons]
        assert FallbackCode.RUNTIME_ADAPTER_FUNC_UNREFERENCED.value in codes

    def test_binding_incomplete(self):
        score = _make_score()
        ph = _make_plan_handle(score, arg_bindings=())
        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(score, ph)
        assert not result.success
        codes = [r.code for r in result.reasons]
        assert FallbackCode.RUNTIME_ADAPTER_BINDING_INCOMPLETE.value in codes

    def test_invalid_edge_producer(self):
        score = _make_score(
            dependencies=(Dependency(producer=99, consumer=1),)
        )
        ph = _make_plan_handle(score)
        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(score, ph)
        assert not result.success
        codes = [r.code for r in result.reasons]
        assert FallbackCode.RUNTIME_ADAPTER_INVALID_EDGE.value in codes

    def test_validate_returns_empty_for_valid(self):
        score = _make_score()
        ph = _make_plan_handle(score)
        adapter = HostBuildGraphRuntimeAdapter()
        reasons = adapter.validate(score, ph)
        assert reasons == ()

    def test_generate_single_task_no_deps(self):
        score = _make_score(
            tasks=(
                Task(task_id=0, func_id=0, core_type="aic",
                     args=("a",), arg_directions=("input",),
                     arg_storage_keys=("param:a",), name="single_op"),
            ),
            dependencies=(),
        )
        ph = _make_plan_handle(score)
        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(score, ph)
        assert result.success
        assert result.plan.task_count() == 1
        assert result.plan.edge_count() == 0

    def test_generate_non_memory_args_no_binding_required(self):
        score = _make_score(
            tasks=(
                Task(task_id=0, func_id=0, core_type="aic",
                     args=("a", "b", "42"),
                     arg_directions=("input", "output", "scalar"),
                     arg_storage_keys=("param:a", "alloc:b", None),
                     name="op"),
            ),
            dependencies=(),
        )
        bindings = (
            RuntimeArgBinding(task_id=0, arg_index=0, storage_key="param:a",
                              direction="input", runtime_handle=0),
            RuntimeArgBinding(task_id=0, arg_index=1, storage_key="alloc:b",
                              direction="output", runtime_handle=1),
        )
        ph = _make_plan_handle(score, arg_bindings=bindings)
        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(score, ph)
        assert result.success


class TestHostBuildGraphPlan:
    def test_empty_plan(self):
        plan = HostBuildGraphPlan()
        assert plan.task_count() == 0
        assert plan.edge_count() == 0

    def test_task_count(self):
        plan = HostBuildGraphPlan(
            tasks=(
                HostBuildGraphTask(task_id=0, func_id=0, core_type="aic"),
                HostBuildGraphTask(task_id=1, func_id=1, core_type="aiv"),
            ),
        )
        assert plan.task_count() == 2


class TestRuntimeAdapterResult:
    def test_accept(self):
        plan = HostBuildGraphPlan()
        result = RuntimeAdapterResult.accept(plan)
        assert result.success
        assert result.plan is plan
        assert result.reasons == ()

    def test_reject(self):
        from sonata.score import FallbackReason
        reason = FallbackReason(code="test", message="test reason")
        result = RuntimeAdapterResult.reject(reason)
        assert not result.success
        assert result.plan is None
        assert len(result.reasons) == 1
