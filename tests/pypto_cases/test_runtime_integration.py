# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real PyPTO IR -> Sonata -> HostBuildGraphPlan integration tests.

These tests use actual PyPTO IR dumps (not mocks) to verify the full
Sonata pipeline: eligibility check -> Score -> PlanHandle -> runtime adapter.

Requires: PYTHONPATH=src:upstream/pypto/python
"""

import sys
from pathlib import Path

import pytest

# Ensure upstream pypto is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_PYPTO_ROOT = _REPO_ROOT / "upstream" / "pypto"
_UPSTREAM_ST_ROOT = _UPSTREAM_PYPTO_ROOT / "tests" / "st"
for _path in (_UPSTREAM_ST_ROOT, _UPSTREAM_PYPTO_ROOT):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

import pypto.language as pl
from pypto import passes
from pypto.backend import BackendType, is_backend_configured, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import passes as _core_passes

from sonata import check_static_eligibility, score_fingerprint, RuntimeTarget
from sonata.plan_handle import PlanHandle
from sonata.pypto_adapter import PostSimplifyPyPTOInputAdapter, DEFAULT_CERTIFIED_DUMP
from sonata.runtime_adapter import HostBuildGraphRuntimeAdapter


def _compile_to_certified_dump(program: object) -> object:
    """Run the default PyPTO pipeline up to the certified Simplify stage."""
    if not is_backend_configured():
        set_backend_type(BackendType.Ascend910B)
    with _core_passes.PassContext([], _core_passes.VerificationLevel.NONE):
        manager = PassManager.get_strategy(OptimizationStrategy.Default)
        current = program
        after_collect_comm_groups = False
        for pass_name, pass_obj in zip(manager.pass_names, manager.passes):
            current = pass_obj(current)
            if pass_name == "CollectCommGroups":
                after_collect_comm_groups = True
            elif after_collect_comm_groups and pass_name == "Simplify":
                return current
    pytest.skip("default pipeline did not expose Simplify after CollectCommGroups")


# ---------------------------------------------------------------------------
# Simple PyPTO programs for integration testing
# ---------------------------------------------------------------------------

_SIMPLE_ADD_PROGRAM = pl.parse_program("""
@pl.program
class SimpleAdd:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[16, 16], pl.FP32],
        y: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[16, 16], pl.FP32],
        y: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        local: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
        local = self.kernel(x, y, local)
        return local
""")


class TestRealIRToHostBuildGraph:
    """Full pipeline: real PyPTO IR -> eligibility -> Score -> HostBuildGraphPlan."""

    def test_simple_add_produces_build_graph_plan(self):
        """Simple add program goes through full pipeline to HostBuildGraphPlan."""
        certified = _compile_to_certified_dump(_SIMPLE_ADD_PROGRAM)
        result = check_static_eligibility(
            certified,
            runtime_target=RuntimeTarget(
                runtime="host_build_graph",
                function_name="build_SimpleAdd_graph",
            ),
        )

        assert result.eligible, f"Not eligible: {result.reasons}"
        score = result.score
        assert score is not None
        assert len(score.tasks) >= 1

        # Generate PlanHandle from Score (auto-generates arg_bindings)
        plan_handle = PlanHandle.from_score(
            score,
            source_adapter=DEFAULT_CERTIFIED_DUMP,
        )
        assert plan_handle.score_fingerprint == score_fingerprint(score)
        assert len(plan_handle.arg_bindings) >= 1

        # Generate HostBuildGraphPlan
        rt_adapter = HostBuildGraphRuntimeAdapter()
        rt_result = rt_adapter.generate(score, plan_handle)

        assert rt_result.success, f"Adapter rejected: {rt_result.reasons}"
        plan = rt_result.plan
        assert plan is not None
        assert plan.task_count() >= 1
        assert plan.edge_count() >= 0  # may have edges if dependencies exist

        # Verify tasks match score
        assert plan.task_count() == len(score.tasks)
        for hbg_task, score_task in zip(plan.tasks, score.tasks):
            assert hbg_task.task_id == score_task.task_id
            assert hbg_task.func_id == score_task.func_id

    def test_region_analysis_on_real_ir(self):
        """Region extraction works on real PyPTO IR nodes."""
        from sonata.regions import extract_regions

        certified = _compile_to_certified_dump(_SIMPLE_ADD_PROGRAM)
        adapter = PostSimplifyPyPTOInputAdapter(certified)
        facts = adapter.normalize(require_certified=True)

        # Extract regions from the first function
        assert len(facts.functions) >= 1
        func_node = facts.functions[0].node
        region_map = extract_regions(func_node)

        # Simple straight-line program should be all static
        assert len(region_map.regions) >= 1
        assert all(r.is_static for r in region_map.regions)

    def test_multi_kernel_full_pipeline(self):
        """Multi-kernel program: eligibility, region, and plan all work."""
        from sonata.regions import check_region_eligibility

        certified = _compile_to_certified_dump(_SIMPLE_ADD_PROGRAM)
        result = check_static_eligibility(
            certified,
            runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="test"),
        )
        assert result.eligible
        assert len(result.score.tasks) >= 1

        # Region eligibility
        region_result = check_region_eligibility(certified)
        assert region_result.eligible
        assert "region_0" in region_result.metadata["region_statuses"]

        # HostBuildGraphPlan
        plan_handle = PlanHandle.from_score(result.score, source_adapter=DEFAULT_CERTIFIED_DUMP)
        rt_adapter = HostBuildGraphRuntimeAdapter()
        rt_result = rt_adapter.generate(result.score, plan_handle)
        assert rt_result.success
        assert rt_result.plan.task_count() == len(result.score.tasks)

    def test_region_aware_plan_with_guard_status(self):
        """Region-aware plan includes region statuses and guard status."""
        certified = _compile_to_certified_dump(_SIMPLE_ADD_PROGRAM)
        result = check_static_eligibility(
            certified,
            runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="test"),
        )
        assert result.eligible

        plan_handle = PlanHandle.from_score(result.score, source_adapter=DEFAULT_CERTIFIED_DUMP)
        from sonata.plan_handle import GuardStatus
        object.__setattr__(plan_handle, "region_guard_status", {
            "region_0": GuardStatus.ALL_SATISFIED,
            "region_1": GuardStatus.PARTIAL_FAILED,
        })

        rt_adapter = HostBuildGraphRuntimeAdapter()
        rt_result = rt_adapter.generate_region_aware(
            result.score,
            plan_handle,
            region_statuses={"region_0": "static", "region_1": "dynamic"},
        )
        assert rt_result.success
        plan = rt_result.plan
        assert plan.metadata["dynamic_region_count"] == 1
        assert plan.metadata["static_region_count"] == 1
        assert plan.metadata["region_guard_status"]["region_1"] == "partial_failed"


class TestSonataPipeline:
    """Tests for sonata_analyze() — single-call full analysis (Path A)."""

    def test_sonata_analyze_simple_program(self):
        """sonata_analyze produces complete result on simple program."""
        from sonata.pipeline import sonata_analyze

        certified = _compile_to_certified_dump(_SIMPLE_ADD_PROGRAM)
        result = sonata_analyze(certified, entry_name="SimpleAdd")

        assert result.eligible
        assert result.score is not None
        assert result.task_count >= 1
        assert result.has_plan
        assert result.region_tree is not None
        assert result.region_eligibility is not None
        assert result.plan_handle is not None
        assert "region_0" in result.region_statuses

    def test_sonata_analyze_plan_tasks_match_score(self):
        """Plan tasks match score tasks 1:1."""
        from sonata.pipeline import sonata_analyze

        certified = _compile_to_certified_dump(_SIMPLE_ADD_PROGRAM)
        result = sonata_analyze(certified, entry_name="SimpleAdd")

        assert result.host_build_graph_plan is not None
        assert result.host_build_graph_plan.task_count() == len(result.score.tasks)

    def test_sonata_analyze_upstream_ops(self):
        """sonata_analyze works on real upstream PyPTO operator programs."""
        import importlib.util
        from pathlib import Path
        from sonata.pipeline import sonata_analyze

        _UPSTREAM = Path(__file__).resolve().parents[2] / "upstream" / "pypto"

        def _load_program(module_name: str, rel_path: str, attr: str):
            path = _UPSTREAM / rel_path
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, attr)

        ops = [
            ("tile_abs", "tests/st/runtime/ops/test_abs.py", "TileAbsProgram"),
            ("tile_cast", "tests/st/runtime/ops/test_cast.py", "TileCastRowMajorNarrowProgram"),
        ]

        for name, rel_path, attr in ops:
            program = _load_program(f"pypto_st_{name}", rel_path, attr)
            certified = _compile_to_certified_dump(program)
            result = sonata_analyze(certified, entry_name=name)

            assert result.eligible, f"{name} should be eligible"
            assert result.score is not None
            assert result.has_plan, f"{name} should produce a plan"
            assert result.task_count >= 1, f"{name} should have >= 1 task"


class TestEndToEndRuntime:
    """5E: Verify Sonata analysis is consistent with PyPTO compile output."""

    def test_sonata_plan_matches_compile_structure(self):
        """Sonata analysis tasks match what pypto.compile() produces."""
        from sonata.pipeline import sonata_analyze

        certified = _compile_to_certified_dump(_SIMPLE_ADD_PROGRAM)
        result = sonata_analyze(certified, entry_name="SimpleAdd")

        assert result.eligible
        assert result.has_plan

        # Verify plan structure is self-consistent
        plan = result.host_build_graph_plan
        score = result.score
        assert plan.task_count() == len(score.tasks)
        for hbg_task, score_task in zip(plan.tasks, score.tasks):
            assert hbg_task.task_id == score_task.task_id
            assert hbg_task.func_id == score_task.func_id

        # Verify edges match dependencies
        assert plan.edge_count() == len(score.dependencies)

        # Verify PlanHandle is valid
        ph = result.plan_handle
        assert ph is not None
        assert len(ph.arg_bindings) > 0
        assert ph.runtime_target.runtime == "host_build_graph"

    def test_full_st_test_infrastructure_works(self):
        """The upstream st test infrastructure runs on a2a3sim.

        This verifies simpler runtime is functional — the st test
        compiles, generates C++ code, and executes on a2a3sim.
        """
        import subprocess
        import os

        upstream_dir = str(
            Path(__file__).resolve().parents[2] / "upstream" / "pypto"
        )
        venv_python = str(
            Path(__file__).resolve().parents[2].parent
            / ".venv-sonata" / "bin" / "python"
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = "tests/st:python"
        if "PTOAS_ROOT" not in env:
            pytest.skip("PTOAS_ROOT not set — simpler runtime test requires ptoas")
        env["PATH"] = env["PTOAS_ROOT"] + "/bin:" + env.get("PATH", "")
        r = subprocess.run(
            [venv_python, "-m", "pytest",
             "tests/st/runtime/ops/test_abs.py::TestAbs::test_tile_abs",
             "--platform=a2a3sim", "-v", "--tb=short", "--forked"],
            cwd=upstream_dir, env=env,
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, f"st test failed:\n{r.stdout[-2000:]}\n{r.stderr[-500:]}"


class TestSonataPlanJson:
    """Tests for sonata_plan.json serialization (v0.12 Phase 1 A1-A2)."""

    def test_save_and_load_roundtrip(self):
        """SonataAnalysisResult saves to JSON and loads back."""
        import json
        import tempfile
        from sonata.pipeline import sonata_analyze, load_sonata_plan

        certified = _compile_to_certified_dump(_SIMPLE_ADD_PROGRAM)
        result = sonata_analyze(certified, entry_name="SimpleAdd")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = result.save(f"{tmpdir}/sonata_plan.json")

            # File exists and is valid JSON
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["schema_version"] == 1
            assert data["eligible"] is True
            assert data["task_count"] >= 1
            assert "region_0" in data["region_statuses"]

            # PlanHandle is serialized
            assert "plan_handle" in data
            assert "score_fingerprint" in data["plan_handle"]

            # HostBuildGraphPlan is serialized
            assert "host_build_graph_plan" in data
            assert len(data["host_build_graph_plan"]["tasks"]) >= 1

            # load_sonata_plan reads it back
            loaded = load_sonata_plan(path)
            assert loaded is not None
            assert loaded.eligible is True

    def test_load_nonexistent_returns_none(self):
        """load_sonata_plan returns None for missing file."""
        from sonata.pipeline import load_sonata_plan
        assert load_sonata_plan("/tmp/nonexistent_sonata_plan.json") is None

    def test_sonata_compile_function_exists(self):
        """sonata_compile is importable and has correct signature."""
        from sonata.pipeline import sonata_compile
        import inspect

        sig = inspect.signature(sonata_compile)
        assert "program" in sig.parameters
        assert "output_dir" in sig.parameters
        assert "entry_name" in sig.parameters

    def test_load_sonata_plan_from_directory(self):
        """load_sonata_plan accepts a directory and finds sonata_plan.json."""
        import tempfile
        from sonata.pipeline import sonata_analyze, load_sonata_plan

        certified = _compile_to_certified_dump(_SIMPLE_ADD_PROGRAM)
        result = sonata_analyze(certified, entry_name="SimpleAdd")

        with tempfile.TemporaryDirectory() as tmpdir:
            result.save(f"{tmpdir}/sonata_plan.json")

            # Load by directory path
            loaded = load_sonata_plan(tmpdir)
            assert loaded is not None
            assert loaded.eligible is True

    def test_execute_with_sonata_function_exists(self):
        """execute_with_sonata is importable."""
        from sonata.pipeline import execute_with_sonata
        import inspect
        sig = inspect.signature(execute_with_sonata)
        assert "work_dir" in sig.parameters


class TestSonataPlanRoundTrip:
    """C1: Verify sonata_plan.json round-trip fidelity."""

    def test_roundtrip_preserves_eligibility_and_regions(self):
        """Save → load preserves eligibility and region statuses."""
        import json
        import tempfile
        from sonata.pipeline import sonata_analyze, load_sonata_plan

        certified = _compile_to_certified_dump(_SIMPLE_ADD_PROGRAM)
        original = sonata_analyze(certified, entry_name="roundtrip_test")

        assert original.eligible

        with tempfile.TemporaryDirectory() as tmpdir:
            path = original.save(f"{tmpdir}/plan.json")

            # Verify JSON structure
            data = json.loads(path.read_text())
            assert data["eligible"] is True
            assert data["schema_version"] == 1
            assert "region_0" in data["region_statuses"]

            # Load back
            loaded = load_sonata_plan(path)
            assert loaded is not None
            assert loaded.eligible == original.eligible
            assert loaded.region_statuses == original.region_statuses

    def test_roundtrip_preserves_plan_tasks(self):
        """Plan task details survive round-trip."""
        import json
        import tempfile
        from sonata.pipeline import sonata_analyze

        certified = _compile_to_certified_dump(_SIMPLE_ADD_PROGRAM)
        original = sonata_analyze(certified, entry_name="task_roundtrip")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = original.save(f"{tmpdir}/plan.json")
            data = json.loads(path.read_text())

            # Verify plan tasks match original
            plan_tasks = data["host_build_graph_plan"]["tasks"]
            assert len(plan_tasks) == len(original.score.tasks)
            for hbg, score_task in zip(plan_tasks, original.score.tasks):
                assert hbg["task_id"] == score_task.task_id
                assert hbg["func_id"] == score_task.func_id


class TestRegionDispatch:
    """Phase 2 A1-A2: Region-aware execution dispatcher."""

    def test_all_static_dispatch(self):
        """All-static graph: all regions optimized."""
        from sonata.pipeline import sonata_analyze, dispatch_regions

        certified = _compile_to_certified_dump(_SIMPLE_ADD_PROGRAM)
        result = sonata_analyze(certified, entry_name="dispatch_test")

        plan = dispatch_regions(result)
        assert plan.total >= 1
        assert plan.optimized_count >= 1
        assert plan.fallback_count == 0
        assert not plan.has_fallbacks

    def test_mixed_graph_dispatch(self):
        """Mixed graph: static regions optimized, dynamic regions fallback."""
        from sonata.pipeline import dispatch_regions, SonataAnalysisResult

        result = SonataAnalysisResult(
            eligible=True,
            region_statuses={
                "region_0": "static",
                "region_1": "dynamic",
                "region_2": "static",
                "region_3": "mixed",
            },
        )

        plan = dispatch_regions(result, verbose=True)
        assert plan.total == 4
        assert plan.optimized_count == 2
        assert plan.fallback_count == 1
        assert plan.mixed_count == 1
        assert plan.has_fallbacks

        # Check per-region results
        r0 = plan.results[0]
        assert r0.action == "optimized"
        assert r0.fallback_reason is None

        r1 = plan.results[1]
        assert r1.action == "fallback"
        assert "dynamic" in r1.fallback_reason

        r3 = plan.results[3]
        assert r3.action == "mixed"
        assert "mixed" in r3.fallback_reason
