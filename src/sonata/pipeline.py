# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Sonata analysis pipeline for PyPTO IR.

Provides ``sonata_analyze()`` — a single-call entry point that runs the full
Sonata analysis suite on a certified post-Simplify IR dump and returns a
structured result containing eligibility, region analysis, Score, PlanHandle,
and HostBuildGraphPlan.

Designed to be called between the PyPTO pass pipeline and codegen (Path A
integration), or standalone for testing and validation.

Usage::

    from sonata.pipeline import sonata_analyze

    result = sonata_analyze(certified_ir)
    if result.eligible:
        print(result.plan.task_count(), "tasks")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .eligibility import check_static_eligibility
from .plan_handle import GuardStatus, PlanHandle
from .pypto_adapter import DEFAULT_CERTIFIED_DUMP, PostSimplifyPyPTOInputAdapter
from .regions import (
    RegionTree,
    build_region_tree,
    check_region_eligibility,
    extract_regions,
)
from .runtime_adapter import (
    HostBuildGraphPlan,
    HostBuildGraphRuntimeAdapter,
    RuntimeAdapterResult,
)
from .score import EligibilityResult, RuntimeTarget, Score


@dataclass(frozen=True)
class SonataAnalysisResult:
    """Complete result of running Sonata analysis on a PyPTO IR dump."""

    # Eligibility
    eligible: bool
    score: Score | None = None
    eligibility_result: EligibilityResult | None = None

    # Region analysis
    region_tree: RegionTree | None = None
    region_eligibility: EligibilityResult | None = None

    # Runtime plan
    plan_handle: PlanHandle | None = None
    host_build_graph_plan: HostBuildGraphPlan | None = None
    adapter_result: RuntimeAdapterResult | None = None

    # Metadata
    region_statuses: dict[str, str] = field(default_factory=dict)
    fallback_reasons: tuple[Any, ...] = ()

    @property
    def task_count(self) -> int:
        if self.host_build_graph_plan is not None:
            return self.host_build_graph_plan.task_count()
        if self.score is not None:
            return len(self.score.tasks)
        return 0

    @property
    def has_plan(self) -> bool:
        return self.host_build_graph_plan is not None


def sonata_analyze(
    certified_ir: object,
    *,
    runtime_target: RuntimeTarget | None = None,
    entry_name: str | None = None,
) -> SonataAnalysisResult:
    """Run the full Sonata analysis suite on a certified IR dump.

    This is the main entry point for Path A integration. It runs:
    1. Eligibility check (whole-graph)
    2. Region extraction and per-region eligibility
    3. Score and PlanHandle generation
    4. HostBuildGraphPlan generation (region-aware)

    Args:
        certified_ir: Post-Simplify IR dump from PyPTO pipeline.
        runtime_target: Runtime target configuration.
        entry_name: Entry function name for naming.

    Returns:
        SonataAnalysisResult with all analysis outputs.
    """
    rt = runtime_target or RuntimeTarget(
        runtime="host_build_graph",
        function_name=entry_name or "graph",
    )

    # Step 1: Whole-graph eligibility
    elig = check_static_eligibility(certified_ir, runtime_target=rt)

    if not elig.eligible:
        return SonataAnalysisResult(
            eligible=False,
            eligibility_result=elig,
            fallback_reasons=elig.reason_details,
        )

    score = elig.score

    # Step 2: Region analysis
    region_map = extract_regions(certified_ir)
    region_tree = build_region_tree(region_map)
    region_elig = check_region_eligibility(certified_ir)

    region_statuses = {}
    if region_elig.metadata:
        region_statuses = region_elig.metadata.get("region_statuses", {})

    # Step 3: PlanHandle
    plan_handle = PlanHandle.from_score(
        score,
        source_adapter=DEFAULT_CERTIFIED_DUMP,
    )

    # Step 4: HostBuildGraphPlan (region-aware)
    rt_adapter = HostBuildGraphRuntimeAdapter()
    rt_result = rt_adapter.generate_region_aware(
        score,
        plan_handle,
        region_statuses=region_statuses,
    )

    return SonataAnalysisResult(
        eligible=True,
        score=score,
        eligibility_result=elig,
        region_tree=region_tree,
        region_eligibility=region_elig,
        plan_handle=plan_handle,
        host_build_graph_plan=rt_result.plan if rt_result.success else None,
        adapter_result=rt_result,
        region_statuses=region_statuses,
    )
