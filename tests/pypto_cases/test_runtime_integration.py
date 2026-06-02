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
