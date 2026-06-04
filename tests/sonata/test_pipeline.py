# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for sonata.pipeline module (v0.12-0.13)."""

import json
import tempfile
from pathlib import Path

import pytest

from sonata.pipeline import (
    DispatchPlan,
    GuardCheckResult,
    RegionDispatchResult,
    SONATA_PLAN_SCHEMA_VERSION,
    SonataAnalysisResult,
    check_guards_at_runtime,
    dispatch_regions,
    load_sonata_plan,
    update_region_guard_status,
)
from sonata.score import DependencyKind, RuntimeTarget, Score, Task
from sonata.guard import ShapeAssumption, GUARD_SEVERITY_HARD, GUARD_SEVERITY_SOFT
from sonata.plan_handle import GuardStatus


def _make_score(assumptions=()):
    rt = RuntimeTarget(runtime="host_build_graph", function_name="test", aicpu_thread_num=1)
    return Score(name="test", runtime_target=rt, tasks=(), dependencies=(),
                 shape_assumptions=tuple(assumptions))


# --- SonataAnalysisResult ---

class TestSonataAnalysisResult:
    def test_eligible_result_properties(self):
        score = _make_score()
        r = SonataAnalysisResult(eligible=True, score=score, region_statuses={"r0": "static"})
        assert r.task_count == 0
        assert not r.has_plan

    def test_ineligible_result(self):
        r = SonataAnalysisResult(eligible=False)
        assert r.task_count == 0
        assert not r.has_plan

    def test_to_dict_schema_version(self):
        r = SonataAnalysisResult(eligible=True, region_statuses={"r0": "static"})
        data = r.to_dict()
        assert data["schema_version"] == SONATA_PLAN_SCHEMA_VERSION
        assert data["eligible"] is True

    def test_to_dict_includes_region_statuses(self):
        r = SonataAnalysisResult(eligible=True, region_statuses={"r0": "static", "r1": "dynamic"})
        data = r.to_dict()
        assert data["region_statuses"]["r0"] == "static"
        assert data["region_statuses"]["r1"] == "dynamic"

    def test_save_and_load_roundtrip(self):
        r = SonataAnalysisResult(eligible=True, region_statuses={"r0": "static"})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = r.save(f"{tmpdir}/plan.json")
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["eligible"] is True
            loaded = load_sonata_plan(path)
            assert loaded is not None
            assert loaded.eligible is True

    def test_save_creates_parent_dirs(self):
        r = SonataAnalysisResult(eligible=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = r.save(f"{tmpdir}/sub/dir/plan.json")
            assert path.exists()

    def test_load_nonexistent(self):
        assert load_sonata_plan("/tmp/nonexistent_sonata_plan.json") is None

    def test_load_from_directory(self):
        r = SonataAnalysisResult(eligible=True, region_statuses={"r0": "static"})
        with tempfile.TemporaryDirectory() as tmpdir:
            r.save(f"{tmpdir}/sonata_plan.json")
            loaded = load_sonata_plan(tmpdir)
            assert loaded is not None
            assert loaded.eligible is True


# --- dispatch_regions ---

class TestDispatchRegions:
    def test_all_static(self):
        r = SonataAnalysisResult(eligible=True, region_statuses={"r0": "static", "r1": "static"})
        plan = dispatch_regions(r)
        assert plan.optimized_count == 2
        assert plan.fallback_count == 0
        assert not plan.has_fallbacks

    def test_all_dynamic(self):
        r = SonataAnalysisResult(eligible=True, region_statuses={"r0": "dynamic", "r1": "dynamic"})
        plan = dispatch_regions(r)
        assert plan.fallback_count == 2
        assert plan.optimized_count == 0
        assert plan.has_fallbacks

    def test_mixed(self):
        r = SonataAnalysisResult(eligible=True, region_statuses={
            "r0": "static", "r1": "dynamic", "r2": "mixed",
        })
        plan = dispatch_regions(r)
        assert plan.optimized_count == 1
        assert plan.fallback_count == 1
        assert plan.mixed_count == 1

    def test_empty_regions(self):
        r = SonataAnalysisResult(eligible=True, region_statuses={})
        plan = dispatch_regions(r)
        assert plan.total == 0

    def test_dispatch_result_fields(self):
        r = SonataAnalysisResult(eligible=True, region_statuses={"r0": "dynamic"})
        plan = dispatch_regions(r)
        result = plan.results[0]
        assert result.region_id == "r0"
        assert result.status == "dynamic"
        assert result.action == "fallback"
        assert "dynamic" in result.fallback_reason

    def test_dispatch_includes_dependency_kinds(self):
        """Dispatch results include dependency kinds from score."""
        from sonata.score import Dependency, DependencyKind
        rt = RuntimeTarget(runtime="host_build_graph", function_name="f", aicpu_thread_num=1)
        score = Score(
            name="test", runtime_target=rt, tasks=(),
            dependencies=(
                Dependency(producer=0, consumer=1, kind=DependencyKind.DATA),
                Dependency(producer=1, consumer=2, kind=DependencyKind.WAR),
            ),
            shape_assumptions=(),
        )
        r = SonataAnalysisResult(eligible=True, score=score, region_statuses={"r0": "static"})
        plan = dispatch_regions(r)
        assert "data" in plan.results[0].dependency_kinds
        assert "war" in plan.results[0].dependency_kinds


# --- check_guards_at_runtime ---

class TestCheckGuards:
    def test_no_shape_assumptions(self):
        score = _make_score()
        r = SonataAnalysisResult(eligible=True, score=score, region_statuses={"r0": "static"})
        results = check_guards_at_runtime(r, {})
        assert results[0].guard_status == "all_satisfied"

    def test_satisfied_guard(self):
        score = _make_score([ShapeAssumption(symbol="x", dims=(32,), severity=GUARD_SEVERITY_HARD)])
        r = SonataAnalysisResult(eligible=True, score=score, region_statuses={"r0": "static"})
        results = check_guards_at_runtime(r, {"x": (32,)})
        assert results[0].guard_status == "all_satisfied"
        assert results[0].violated_guards == ()

    def test_hard_violation(self):
        score = _make_score([ShapeAssumption(symbol="x", dims=(32,), severity=GUARD_SEVERITY_HARD)])
        r = SonataAnalysisResult(eligible=True, score=score, region_statuses={"r0": "static"})
        results = check_guards_at_runtime(r, {"x": (64,)})
        assert results[0].guard_status == "all_failed"
        assert "x" in results[0].violated_guards

    def test_soft_violation(self):
        score = _make_score([ShapeAssumption(symbol="x", dims=(32,), severity=GUARD_SEVERITY_SOFT)])
        r = SonataAnalysisResult(eligible=True, score=score, region_statuses={"r0": "static"})
        results = check_guards_at_runtime(r, {"x": (64,)})
        assert results[0].guard_status == "stale"

    def test_multiple_guards_mixed(self):
        score = _make_score([
            ShapeAssumption(symbol="x", dims=(32,), severity=GUARD_SEVERITY_HARD),
            ShapeAssumption(symbol="y", dims=(16,), severity=GUARD_SEVERITY_SOFT),
        ])
        r = SonataAnalysisResult(eligible=True, score=score, region_statuses={"r0": "static"})
        # x satisfied, y violated (soft) → stale
        results = check_guards_at_runtime(r, {"x": (32,), "y": (99,)})
        assert results[0].guard_status == "stale"
        assert "y" in results[0].violated_guards
        assert "x" not in results[0].violated_guards

    def test_no_score(self):
        r = SonataAnalysisResult(eligible=True, score=None, region_statuses={"r0": "static"})
        results = check_guards_at_runtime(r, {})
        assert results[0].guard_status == "all_satisfied"


# --- update_region_guard_status ---

class TestUpdateRegionGuardStatus:
    def test_all_satisfied(self):
        results = (GuardCheckResult(region_id="r0", guard_status="all_satisfied"),)
        status = update_region_guard_status(None, results)
        assert status["r0"] == GuardStatus.ALL_SATISFIED

    def test_mixed_statuses(self):
        results = (
            GuardCheckResult(region_id="r0", guard_status="all_satisfied"),
            GuardCheckResult(region_id="r1", guard_status="partial_failed"),
            GuardCheckResult(region_id="r2", guard_status="all_failed"),
        )
        status = update_region_guard_status(None, results)
        assert status["r0"] == GuardStatus.ALL_SATISFIED
        assert status["r1"] == GuardStatus.PARTIAL_FAILED
        assert status["r2"] == GuardStatus.ALL_FAILED

    def test_invalid_status_defaults_to_all_failed(self):
        results = (GuardCheckResult(region_id="r0", guard_status="unknown_value"),)
        status = update_region_guard_status(None, results)
        assert status["r0"] == GuardStatus.ALL_FAILED

    def test_empty_results(self):
        status = update_region_guard_status(None, ())
        assert status == {}


# --- DependencyKind integration ---

class TestDependencyKindIntegration:
    def test_dependency_kind_in_score_to_dict(self):
        """DependencyKind serializes correctly in full Score context."""
        from sonata.serialization import score_to_dict
        from sonata.score import Dependency
        rt = RuntimeTarget(runtime="host_build_graph", function_name="f", aicpu_thread_num=1)
        dep = Dependency(producer=0, consumer=1, kind=DependencyKind.STORAGE)
        score = Score(name="test", runtime_target=rt, tasks=(),
                      dependencies=(dep,), shape_assumptions=())
        data = score_to_dict(score)
        assert data["dependencies"][0]["kind"] == "storage"


# --- compute_scheduling_instructions ---

class TestSchedulingInstructions:
    def test_static_region_optimized(self):
        from sonata.pipeline import compute_scheduling_instructions, DispatchPlan, RegionDispatchResult
        dispatch = DispatchPlan(
            results=(RegionDispatchResult(region_id="r0", status="static", action="optimized"),),
            optimized_count=1,
        )
        inst = compute_scheduling_instructions(dispatch, base_block_dim=32)
        assert inst[0].block_dim == 32
        assert "optimized" in inst[0].reason

    def test_dynamic_region_fallback(self):
        from sonata.pipeline import compute_scheduling_instructions, DispatchPlan, RegionDispatchResult
        dispatch = DispatchPlan(
            results=(RegionDispatchResult(region_id="r0", status="dynamic", action="fallback"),),
            fallback_count=1,
        )
        inst = compute_scheduling_instructions(dispatch, fallback_block_dim=1)
        assert inst[0].block_dim == 1
        assert "fallback" in inst[0].reason

    def test_mixed_region_conservative(self):
        from sonata.pipeline import compute_scheduling_instructions, DispatchPlan, RegionDispatchResult
        dispatch = DispatchPlan(
            results=(RegionDispatchResult(region_id="r0", status="mixed", action="mixed"),),
            mixed_count=1,
        )
        inst = compute_scheduling_instructions(dispatch, base_block_dim=32)
        assert inst[0].block_dim == 16  # half of base
        assert "conservative" in inst[0].reason

    def test_end_to_end_scheduling(self):
        """Full pipeline: analyze → dispatch → scheduling instructions."""
        from sonata.pipeline import sonata_analyze, dispatch_regions, compute_scheduling_instructions
        from sonata.score import Score, RuntimeTarget, Dependency, DependencyKind

        rt = RuntimeTarget(runtime="host_build_graph", function_name="f", aicpu_thread_num=1)
        score = Score(
            name="test", runtime_target=rt,
            tasks=(), dependencies=(
                Dependency(producer=0, consumer=1, kind=DependencyKind.DATA),
            ),
            shape_assumptions=(),
        )
        result = SonataAnalysisResult(
            eligible=True, score=score,
            region_statuses={"r0": "static", "r1": "dynamic", "r2": "mixed"},
        )

        dispatch = dispatch_regions(result)
        instructions = compute_scheduling_instructions(dispatch, base_block_dim=64)

        assert len(instructions) == 3
        assert instructions[0].block_dim == 64   # static → base
        assert instructions[1].block_dim == 1    # dynamic → fallback
        assert instructions[2].block_dim == 32   # mixed → half

    def test_scheduling_with_custom_dims(self):
        """Custom base/fallback dims are respected."""
        from sonata.pipeline import compute_scheduling_instructions, DispatchPlan, RegionDispatchResult
        dispatch = DispatchPlan(
            results=(
                RegionDispatchResult(region_id="r0", status="static", action="optimized"),
                RegionDispatchResult(region_id="r1", status="dynamic", action="fallback"),
            ),
            optimized_count=1, fallback_count=1,
        )
        inst = compute_scheduling_instructions(dispatch, base_block_dim=16, fallback_block_dim=4)
        assert inst[0].block_dim == 16
        assert inst[1].block_dim == 4


class TestMemoryPlanIntegration:
    """v0.18 Phase 1 C1: sonata_analyze outputs MemoryPlan."""

    def test_result_has_memory_plan(self):
        """SonataAnalysisResult with score produces a memory plan."""
        import warnings
        from sonata.score import Score, RuntimeTarget, Task
        from sonata.liveness import compute_lifetimes
        from sonata.memory_plan import plan_memory
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            score = Score(
                name="test",
                runtime_target=RuntimeTarget(),
                tasks=(
                    Task(task_id=0, func_id=0, core_type="aic",
                         args=("x", "y"), arg_directions=("input", "output"),
                         arg_storage_keys=("buf:x", "buf:y")),
                    Task(task_id=1, func_id=1, core_type="aiv",
                         args=("y", "z"), arg_directions=("input", "output"),
                         arg_storage_keys=("buf:y", "buf:z")),
                ),
            )
        lifetimes = compute_lifetimes(score.tasks)
        buffer_sizes = {lt.storage_key: 1024 for lt in lifetimes}
        mp = plan_memory(lifetimes, buffer_sizes)
        result = SonataAnalysisResult(
            eligible=True, score=score, memory_plan=mp,
        )
        assert result.memory_plan is not None
        assert result.memory_plan.peak_memory >= 0

    def test_memory_plan_in_to_dict(self):
        """memory_plan appears in to_dict() output."""
        import warnings
        from sonata.score import Score, RuntimeTarget, Task
        from sonata.liveness import compute_lifetimes
        from sonata.memory_plan import plan_memory
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            score = Score(
                name="test",
                runtime_target=RuntimeTarget(),
                tasks=(
                    Task(task_id=0, func_id=0, core_type="aic",
                         args=("x",), arg_directions=("input",),
                         arg_storage_keys=("buf:x",)),
                ),
            )
        lifetimes = compute_lifetimes(score.tasks)
        buffer_sizes = {lt.storage_key: 1024 for lt in lifetimes}
        mp = plan_memory(lifetimes, buffer_sizes)
        result = SonataAnalysisResult(
            eligible=True, score=score, memory_plan=mp,
        )
        d = result.to_dict()
        assert "memory_plan" in d
        assert "peak_memory" in d["memory_plan"]
        assert "allocations" in d["memory_plan"]

    def test_no_memory_plan_when_none(self):
        """No memory_plan in to_dict() when memory_plan is None."""
        import warnings
        from sonata.score import Score, RuntimeTarget, Task
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            score = Score(
                name="test",
                runtime_target=RuntimeTarget(),
                tasks=(Task(task_id=0, func_id=0, core_type="aic"),),
            )
        result = SonataAnalysisResult(eligible=True, score=score)
        d = result.to_dict()
        assert "memory_plan" not in d


class TestEdgeCases:
    """v0.19 Phase 2 B1: Edge case tests for pipeline robustness."""

    def test_empty_score_no_tasks(self):
        """Score with 0 tasks → memory_plan is None."""
        import warnings
        from sonata.score import Score, RuntimeTarget
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            score = Score(name="empty", runtime_target=RuntimeTarget(), tasks=())
        result = SonataAnalysisResult(eligible=True, score=score)
        d = result.to_dict()
        assert d["task_count"] == 0
        assert "memory_plan" not in d

    def test_empty_score_validation(self):
        """Score with 0 tasks passes validation."""
        import warnings
        from sonata.score import Score, RuntimeTarget
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            score = Score(name="empty", runtime_target=RuntimeTarget(), tasks=())
        validation = score.validate()
        assert validation.eligible

    def test_none_score_result(self):
        """SonataAnalysisResult with score=None → no crash in to_dict."""
        result = SonataAnalysisResult(eligible=False, score=None)
        d = result.to_dict()
        assert d["eligible"] is False
        assert d["task_count"] == 0
        assert "memory_plan" not in d
        assert "guard_stats" not in d
        assert "score" not in d

    def test_empty_dependencies(self):
        """Score with tasks but no dependencies → valid."""
        import warnings
        from sonata.score import Score, RuntimeTarget, Task
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            score = Score(
                name="single",
                runtime_target=RuntimeTarget(),
                tasks=(Task(task_id=0, func_id=0, core_type="aic"),),
                dependencies=(),
            )
        assert score.dependency_count() == 0
        validation = score.validate()
        assert validation.eligible

    def test_empty_region_statuses(self):
        """SonataAnalysisResult with no regions → dispatch returns empty."""
        result = SonataAnalysisResult(eligible=True, region_statuses={})
        dispatch = dispatch_regions(result)
        assert dispatch.total == 0
        assert dispatch.optimized_count == 0


class TestExecuteWithSonataGuardChecks:
    """v0.21 Phase 1 A2: execute_with_sonata() guard check integration."""

    def test_load_sonata_plan_restores_shape_assumptions(self):
        """load_sonata_plan() reconstructs Score with shape_assumptions."""
        import json
        import tempfile
        from pathlib import Path
        from sonata.pipeline import load_sonata_plan

        with tempfile.TemporaryDirectory() as tmpdir:
            plan = {
                "eligible": True,
                "region_statuses": {"region_0": "static"},
                "score": {
                    "name": "test",
                    "shape_assumptions": [
                        {"symbol": "batch", "dims": [32], "severity": "hard"},
                        {"symbol": "seq", "dims": [128], "severity": "soft"},
                    ],
                },
            }
            (Path(tmpdir) / "sonata_plan.json").write_text(json.dumps(plan))
            result = load_sonata_plan(tmpdir)
            assert result is not None
            assert result.score is not None
            assert len(result.score.shape_assumptions) == 2
            assert result.score.shape_assumptions[0].symbol == "batch"

    def test_load_sonata_plan_no_score(self):
        """load_sonata_plan() with no score section → score is None."""
        import json
        import tempfile
        from pathlib import Path
        from sonata.pipeline import load_sonata_plan

        with tempfile.TemporaryDirectory() as tmpdir:
            plan = {"eligible": True, "region_statuses": {"region_0": "static"}}
            (Path(tmpdir) / "sonata_plan.json").write_text(json.dumps(plan))
            result = load_sonata_plan(tmpdir)
            assert result is not None
            assert result.score is None

    def test_guard_check_uses_restored_assumptions(self):
        """Guard check works with shape_assumptions from load_sonata_plan."""
        import json
        import tempfile
        import warnings
        from pathlib import Path
        from sonata.pipeline import load_sonata_plan, check_guards_at_runtime

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with tempfile.TemporaryDirectory() as tmpdir:
                plan = {
                    "eligible": True,
                    "region_statuses": {"region_0": "static"},
                    "score": {
                        "name": "test",
                        "shape_assumptions": [
                            {"symbol": "batch", "dims": [32], "severity": "soft"},
                        ],
                    },
                }
                (Path(tmpdir) / "sonata_plan.json").write_text(json.dumps(plan))
                loaded = load_sonata_plan(tmpdir)
                assert loaded.score is not None

                # Guard check with matching values → all_satisfied
                results = check_guards_at_runtime(loaded, {"batch": [32]})
                assert results[0].guard_status == "all_satisfied"

                # Guard check with mismatched values → stale (soft guard)
                results = check_guards_at_runtime(loaded, {"batch": [64]})
                assert results[0].guard_status == "stale"

    def test_guard_check_hard_guard_all_failed(self):
        """Hard guard violation → all_failed."""
        import json
        import tempfile
        import warnings
        from pathlib import Path
        from sonata.pipeline import load_sonata_plan, check_guards_at_runtime

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with tempfile.TemporaryDirectory() as tmpdir:
                plan = {
                    "eligible": True,
                    "region_statuses": {"region_0": "static"},
                    "score": {
                        "name": "test",
                        "shape_assumptions": [
                            {"symbol": "batch", "dims": [32], "severity": "hard"},
                        ],
                    },
                }
                (Path(tmpdir) / "sonata_plan.json").write_text(json.dumps(plan))
                loaded = load_sonata_plan(tmpdir)
                results = check_guards_at_runtime(loaded, {"batch": [64]})
                assert results[0].guard_status == "all_failed"
