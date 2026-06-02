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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .eligibility import check_static_eligibility
from .plan_handle import PlanHandle
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

SONATA_PLAN_SCHEMA_VERSION = 1


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        v0.12 Phase 1 A1-A2: sonata_plan.json schema.
        """
        from .serialization import plan_handle_to_dict, score_to_dict

        data: dict[str, Any] = {
            "schema_version": SONATA_PLAN_SCHEMA_VERSION,
            "eligible": self.eligible,
            "task_count": self.task_count,
            "region_statuses": self.region_statuses,
        }

        if self.score is not None:
            data["score"] = score_to_dict(self.score)

        if self.plan_handle is not None:
            data["plan_handle"] = plan_handle_to_dict(self.plan_handle)

        if self.host_build_graph_plan is not None:
            plan = self.host_build_graph_plan
            data["host_build_graph_plan"] = {
                "tasks": [
                    {"task_id": t.task_id, "func_id": t.func_id,
                     "core_type": t.core_type, "name": t.name}
                    for t in plan.tasks
                ],
                "edges": [
                    {"producer": e.producer, "consumer": e.consumer}
                    for e in plan.edges
                ],
                "metadata": plan.metadata,
            }

        if self.fallback_reasons:
            data["fallback_reasons"] = [
                {"code": r.code, "message": r.message, "severity": r.severity}
                for r in self.fallback_reasons
                if hasattr(r, "code")
            ]

        return data

    def save(self, path: str | Path) -> Path:
        """Write sonata_plan.json to the given path.

        Args:
            path: Output file path (typically ``<work_dir>/sonata_plan.json``).

        Returns:
            The resolved Path that was written.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return p


def load_sonata_plan(path: str | Path) -> SonataAnalysisResult | None:
    """Load a SonataAnalysisResult from a sonata_plan.json file.

    If *path* is a directory, looks for ``sonata_plan.json`` inside it.
    Returns None if the file does not exist.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "sonata_plan.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return SonataAnalysisResult(
        eligible=data.get("eligible", False),
        region_statuses=data.get("region_statuses", {}),
    )


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


def sonata_compile(
    program: object,
    output_dir: str | Path | None = None,
    *,
    entry_name: str | None = None,
) -> tuple[Any, SonataAnalysisResult]:
    """Compile a PyPTO program and run Sonata analysis.

    v0.12 Phase 1 A3: Wrapper that runs ``pypto.compile()`` then
    ``sonata_analyze()``, writing ``sonata_plan.json`` alongside
    compiled artifacts.

    Args:
        program: A ``@pl.program`` class or ``ir.Program``.
        output_dir: Output directory for compiled artifacts.
        entry_name: Entry function name for Sonata analysis.

    Returns:
        ``(compiled_program, sonata_result)``.
    """
    from pypto import ir as _ir

    compiled = _ir.compile(program, output_dir=output_dir)

    from pypto.backend import BackendType, is_backend_configured, set_backend_type
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager
    from pypto.pypto_core import passes as _core_passes

    if not is_backend_configured():
        set_backend_type(BackendType.Ascend910B)

    with _core_passes.PassContext([], _core_passes.VerificationLevel.NONE):
        manager = PassManager.get_strategy(OptimizationStrategy.Default)
        current = program
        after_ccg = False
        certified_ir = None
        for pname, pobj in zip(manager.pass_names, manager.passes):
            current = pobj(current)
            if pname == "CollectCommGroups":
                after_ccg = True
            elif after_ccg and pname == "Simplify":
                certified_ir = current
                break

    if certified_ir is None:
        return compiled, SonataAnalysisResult(eligible=False)

    result = sonata_analyze(certified_ir, entry_name=entry_name)

    work_dir = Path(compiled._output_dir) if hasattr(compiled, "_output_dir") else None
    if work_dir is not None and result.eligible:
        result.save(work_dir / "sonata_plan.json")

    return compiled, result


def execute_with_sonata(
    work_dir: str | Path,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, SonataAnalysisResult | None]:
    """Execute a compiled program with Sonata plan awareness.

    v0.12 Phase 1 B2: Wrapper around ``pypto.runtime.execute_compiled``
    that reads the Sonata plan from *work_dir* before execution and
    returns it alongside the execution result.

    If no ``sonata_plan.json`` exists in *work_dir*, the plan is None
    and execution proceeds normally (backward compatible).

    Args:
        work_dir: Compiled artifacts directory (contains ``kernel_config.py``
            and optionally ``sonata_plan.json``).
        *args, **kwargs: Forwarded to ``pypto.runtime.execute_compiled``.

    Returns:
        ``(execute_result, sonata_plan)``.
    """
    from pypto.runtime.runner import execute_compiled

    plan = load_sonata_plan(work_dir)

    if plan is not None:
        import logging
        log = logging.getLogger("sonata")
        log.info(
            "Sonata plan loaded: eligible=%s, tasks=%d, regions=%s",
            plan.eligible, plan.task_count, list(plan.region_statuses.keys()),
        )

    execute_compiled(work_dir, *args, **kwargs)
    return None, plan


# ---------------------------------------------------------------------------
# v0.12 Phase 2: Region-Aware Runtime
# ---------------------------------------------------------------------------

import logging as _logging

_region_log = _logging.getLogger("sonata.region_dispatch")


@dataclass(frozen=True)
class RegionDispatchResult:
    """Result of region-aware execution dispatch."""

    region_id: str
    status: str  # "static", "dynamic", "mixed"
    action: str  # "optimized", "fallback", "mixed"
    fallback_reason: str | None = None


@dataclass(frozen=True)
class DispatchPlan:
    """Execution plan produced by the region-aware dispatcher."""

    results: tuple[RegionDispatchResult, ...]
    optimized_count: int = 0
    fallback_count: int = 0
    mixed_count: int = 0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def has_fallbacks(self) -> bool:
        return self.fallback_count > 0


def dispatch_regions(
    sonata_result: SonataAnalysisResult,
    *,
    verbose: bool = False,
) -> DispatchPlan:
    """Dispatch execution strategy per region based on Sonata analysis.

    v0.12 Phase 2 A1: Region-aware execution dispatcher.

    For each region in the Sonata analysis:
    - static: mark as optimized (use Sonata's plan)
    - dynamic: mark as fallback (use original PyPTO runtime path)
    - mixed: mark as mixed (static children optimized, dynamic children fallback)

    Args:
        sonata_result: Result from sonata_analyze().
        verbose: If True, log each region's dispatch decision.

    Returns:
        DispatchPlan with per-region dispatch results.
    """
    results: list[RegionDispatchResult] = []

    for region_id, status in sonata_result.region_statuses.items():
        if status == "static":
            action = "optimized"
            reason = None
        elif status == "dynamic":
            action = "fallback"
            reason = "dynamic control flow region"
        else:  # mixed
            action = "mixed"
            reason = "mixed static/dynamic subtree"

        results.append(RegionDispatchResult(
            region_id=region_id,
            status=status,
            action=action,
            fallback_reason=reason,
        ))

        if verbose:
            _region_log.info(
                "[DISPATCH] %s: status=%s → action=%s%s",
                region_id, status, action,
                f" (reason: {reason})" if reason else "",
            )

    optimized = sum(1 for r in results if r.action == "optimized")
    fallback = sum(1 for r in results if r.action == "fallback")
    mixed = sum(1 for r in results if r.action == "mixed")

    plan = DispatchPlan(
        results=tuple(results),
        optimized_count=optimized,
        fallback_count=fallback,
        mixed_count=mixed,
    )

    if verbose:
        _region_log.info(
            "[DISPATCH SUMMARY] total=%d, optimized=%d, fallback=%d, mixed=%d",
            plan.total, optimized, fallback, mixed,
        )

    return plan
