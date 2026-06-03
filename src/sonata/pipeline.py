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

# Module-level certified IR cache: program id → certified IR
# Avoids re-running the pass pipeline when the same program is analyzed
# multiple times (e.g. in sonata_compile + standalone analysis).
_certified_ir_cache: dict[int, Any] = {}


def _clear_ir_cache() -> None:
    """Clear the certified IR cache.

    Called on HARD guard violation to force re-analysis on next execution.
    """
    _certified_ir_cache.clear()


def _extract_certified_ir(program: object) -> Any | None:
    """Extract certified IR (post-Simplify) from a program.

    Uses a module-level cache keyed by program id to avoid re-running
    the pass pipeline for the same program.
    """
    prog_id = id(program)
    if prog_id in _certified_ir_cache:
        return _certified_ir_cache[prog_id]

    from pypto.ir.pass_manager import OptimizationStrategy, PassManager
    from pypto.pypto_core import passes as _core_passes
    from pypto.backend import BackendType, is_backend_configured, set_backend_type

    if not is_backend_configured():
        set_backend_type(BackendType.Ascend910B)

    certified_ir = None
    try:
        with _core_passes.PassContext([], _core_passes.VerificationLevel.NONE):
            manager = PassManager.get_strategy(OptimizationStrategy.Default)
            current = program
            after_ccg = False
            for pname, pobj in zip(manager.pass_names, manager.passes):
                current = pobj(current)
                if pname == "CollectCommGroups":
                    after_ccg = True
                elif after_ccg and pname == "Simplify":
                    certified_ir = current
                    break
    except Exception:
        certified_ir = None

    _certified_ir_cache[prog_id] = certified_ir
    return certified_ir

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

        # Include dependency kind summary when available
        if self.score is not None and self.score.dependencies:
            dep_kinds: dict[str, int] = {}
            for dep in self.score.dependencies:
                kind_val = getattr(dep.kind, 'value', dep.kind)
                dep_kinds[kind_val] = dep_kinds.get(kind_val, 0) + 1
            data["dependency_kinds"] = dep_kinds

        # Include guard statistics when available (v0.17 Phase 2 A1)
        if self.score is not None and self.score.shape_assumptions:
            assumptions = self.score.shape_assumptions
            unique_symbols = {a.symbol for a in assumptions}
            count = len(assumptions)
            n_symbols = len(unique_symbols)
            density = round(count / n_symbols, 2) if n_symbols > 0 else 0.0
            data["guard_stats"] = {
                "shape_assumption_count": count,
                "unique_symbols": n_symbols,
                "guard_density": density,
            }
            if density > 8:
                data.setdefault("warnings", []).append(
                    f"guard_density={density} exceeds TorchDynamo reference threshold (8)"
                )

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

    # Step 1b: Fallback to region-based eligibility when static check fails
    # This handles both SPMD programs and Orchestration programs with control flow
    if not elig.eligible:
        region_elig = check_region_eligibility(certified_ir)
        if region_elig.eligible:
            elig = region_elig
            _region_log.info(
                "[SONATA] Region-based eligibility: static regions found"
            )

    if not elig.eligible:
        return SonataAnalysisResult(
            eligible=False,
            eligibility_result=elig,
            fallback_reasons=elig.reason_details,
        )

    score = elig.score

    # For region-based eligibility, score may be in per_region_scores metadata
    if score is None and elig.metadata:
        per_region = elig.metadata.get("per_region_scores", {})
        if per_region:
            # Use the first region's score as representative
            score = next(iter(per_region.values()))

    # If score has no tasks (region placeholder), extract real tasks from IR
    if score is not None and len(score.tasks) == 0:
        from .pypto_adapter import PostSimplifyPyPTOInputAdapter
        from .eligibility import _tasks_from_facts
        adapter = PostSimplifyPyPTOInputAdapter(certified_ir)
        try:
            facts = adapter.normalize(require_certified=False)
            if facts.functions:
                real_tasks = _tasks_from_facts(facts.functions)
                if len(real_tasks) > 0:
                    from .dependencies import build_dependencies
                    score = Score(
                        name=score.name or "extracted",
                        runtime_target=rt,
                        tasks=real_tasks,
                        dependencies=build_dependencies(real_tasks),
                        shape_assumptions=score.shape_assumptions,
                        metadata=score.metadata,
                    )
                    _region_log.info(
                        "[SONATA] Extracted %d tasks from IR for region-based eligibility",
                        len(score.tasks),
                    )
        except Exception:
            pass  # keep placeholder score

    if score is None:
        return SonataAnalysisResult(
            eligible=False,
            eligibility_result=elig,
            fallback_reasons=elig.reason_details,
        )

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
    from pypto.pypto_core import passes as _core_passes
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager

    compiled = _ir.compile(program, output_dir=output_dir)

    # Extract certified IR using cached helper (avoids pipeline replay)
    certified_ir = _extract_certified_ir(program)

    if certified_ir is None:
        return compiled, SonataAnalysisResult(eligible=False)

    result = sonata_analyze(certified_ir, entry_name=entry_name)

    work_dir = Path(compiled._output_dir) if hasattr(compiled, "_output_dir") else None
    if work_dir is not None and result.eligible:
        result.save(work_dir / "sonata_plan.json")
        write_memory_hints(result, work_dir)

    return compiled, result


def write_memory_hints(result: SonataAnalysisResult, work_dir: Path) -> Path | None:
    """Write sonata_memory_hint.json alongside compiled artifacts.

    v0.15 Phase 1 A2: Memory offset injection.

    Produces a JSON file with buffer offset hints from Sonata's analysis.
    The simpler runtime can optionally read these hints for memory layout.

    Args:
        result: Sonata analysis result.
        work_dir: Compiled artifacts directory.

    Returns:
        Path to written file, or None if no hints to write.
    """
    if result.plan_handle is None:
        return None

    hints: dict[str, Any] = {
        "schema_version": 1,
        "eligible": result.eligible,
        "task_count": result.task_count,
        "region_count": len(result.region_statuses),
    }

    # Include dependency kind summary if available
    if result.score is not None and result.score.dependencies:
        dep_kinds: dict[str, int] = {}
        for dep in result.score.dependencies:
            kind_val = getattr(dep.kind, "value", dep.kind)
            dep_kinds[kind_val] = dep_kinds.get(kind_val, 0) + 1
        hints["dependency_kinds"] = dep_kinds

    # Include region dispatch hints
    dispatch = dispatch_regions(result)
    hints["dispatch"] = {
        "optimized_count": dispatch.optimized_count,
        "fallback_count": dispatch.fallback_count,
        "mixed_count": dispatch.mixed_count,
    }

    path = work_dir / "sonata_memory_hint.json"
    path.write_text(json.dumps(hints, indent=2, sort_keys=True))
    _region_log.info("[SONATA] memory hints written: %s", path)
    return path


def execute_with_sonata(
    work_dir: str | Path,
    *args: Any,
    runtime_values: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[Any, SonataAnalysisResult | None]:
    """Execute a compiled program with Sonata plan awareness.

    Uses Sonata analysis to influence simpler runtime behavior:
    - Computes suggested block_dim from task count (Sonata-informed)
    - Runs guard checks before execution
    - Logs dispatch decisions
    - On ALL_FAILED guard: skips execution

    Does NOT modify simpler runtime code. All influence is via
    execute_compiled's public parameters (block_dim, aicpu_thread_num).

    Args:
        work_dir: Compiled artifacts directory.
        runtime_values: Tensor shapes/values for guard checking.
        *args, **kwargs: Forwarded to ``pypto.runtime.execute_compiled``.

    Returns:
        ``(execute_result, sonata_result)``.
    """
    from pypto.runtime.runner import execute_compiled

    plan = load_sonata_plan(work_dir)

    if plan is not None and plan.eligible:
        _region_log.info(
            "[SONATA] Plan loaded: eligible=%s, tasks=%d, regions=%s",
            plan.eligible, plan.task_count, list(plan.region_statuses.keys()),
        )

        # Pre-execution: guard check
        if runtime_values and plan.score is not None:
            guard_results = check_guards_at_runtime(plan, runtime_values)
            hard_failed = any(gr.guard_status == "all_failed" for gr in guard_results)
            has_stale = any(gr.guard_status == "stale" for gr in guard_results)
            if hard_failed:
                _clear_ir_cache()
                _region_log.error(
                    "[SONATA] HARD guard violation — cleared IR cache, skipping execution"
                )
                return None, plan
            if has_stale:
                _region_log.warning(
                    "[SONATA] STALE guard — plan handle invalid, Score still valid"
                )

            # Update guard status in sonata_plan.json
            guard_status = update_region_guard_status(plan.plan_handle, guard_results)
            if guard_status:
                plan_path = Path(work_dir) / "sonata_plan.json"
                if plan_path.exists():
                    import json as _json
                    data = _json.loads(plan_path.read_text())
                    data["runtime_guard_status"] = {
                        k: v.value for k, v in guard_status.items()
                    }
                    plan_path.write_text(_json.dumps(data, indent=2, sort_keys=True))
                    _region_log.info("[SONATA] Updated guard status in sonata_plan.json")

        # Pre-execution: region dispatch
        dispatch = dispatch_regions(plan)
        _region_log.info(
            "[SONATA] Dispatch: %d optimized, %d fallback, %d mixed",
            dispatch.optimized_count, dispatch.fallback_count, dispatch.mixed_count,
        )

        # Influence runtime parameters based on analysis
        if plan.task_count > 0 and "block_dim" not in kwargs:
            instructions = compute_scheduling_instructions(dispatch)
            if instructions:
                # Use the first instruction's block_dim
                # (for single-region programs this is the optimized value)
                suggested_block_dim = instructions[0].block_dim
                kwargs["block_dim"] = suggested_block_dim
                _region_log.info(
                    "[SONATA] Setting block_dim=%d (%s)",
                    suggested_block_dim, instructions[0].reason,
                )

    result = execute_compiled(work_dir, *args, **kwargs)
    return result, plan


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
    dependency_kinds: tuple[str, ...] = ()  # DependencyKind values for this region


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

    # Extract dependency kinds from score for dispatch context
    dep_kinds: set[str] = set()
    if sonata_result.score is not None:
        for dep in sonata_result.score.dependencies:
            kind_val = getattr(dep.kind, 'value', dep.kind)
            dep_kinds.add(kind_val)

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
            dependency_kinds=tuple(sorted(dep_kinds)),
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


@dataclass(frozen=True)
class SchedulingInstruction:
    """Scheduling hint derived from region dispatch."""

    region_id: str
    block_dim: int
    reason: str


def compute_scheduling_instructions(
    dispatch: DispatchPlan,
    *,
    base_block_dim: int = 32,
    fallback_block_dim: int = 1,
) -> tuple[SchedulingInstruction, ...]:
    """Generate scheduling instructions from dispatch results.

    v0.15 Phase 2 A1: Region-aware scheduling.

    - static regions → base_block_dim (optimized)
    - dynamic regions → fallback_block_dim (conservative)
    - mixed regions → half of base_block_dim

    Args:
        dispatch: DispatchPlan from dispatch_regions().
        base_block_dim: Block dim for optimized static regions.
        fallback_block_dim: Block dim for dynamic fallback regions.

    Returns:
        Tuple of SchedulingInstruction, one per region.
    """
    instructions: list[SchedulingInstruction] = []
    for result in dispatch.results:
        if result.action == "optimized":
            bd = base_block_dim
            reason = "static region — optimized"
        elif result.action == "fallback":
            bd = fallback_block_dim
            reason = "dynamic region — fallback"
        else:
            bd = max(base_block_dim // 2, 1)
            reason = "mixed region — conservative"
        instructions.append(SchedulingInstruction(
            region_id=result.region_id,
            block_dim=bd,
            reason=reason,
        ))
    return tuple(instructions)


@dataclass(frozen=True)
class GuardDetail:
    """Per-guard evaluation detail (v0.17 Phase 2 B2)."""

    symbol: str
    satisfied: bool
    severity: str  # "soft" or "hard"


@dataclass(frozen=True)
class GuardCheckResult:
    """Result of runtime guard checking.

    v0.17 Phase 2 B2: Enhanced with per-guard details and STALE semantics.
    """

    region_id: str
    guard_status: str  # "all_satisfied", "partial_failed", "all_failed", "stale"
    violated_guards: tuple[str, ...] = ()
    guard_details: tuple[GuardDetail, ...] = ()


def check_guards_at_runtime(
    sonata_result: SonataAnalysisResult,
    runtime_values: dict[str, Any],
    *,
    verbose: bool = False,
) -> tuple[GuardCheckResult, ...]:
    """Check guard conditions for each region before execution.

    v0.12 Phase 2 B1: Runtime guard checker.

    Evaluates shape assumptions (which are GuardConditions) from each
    region's Score against ``runtime_values``. Returns per-region
    guard status.

    Args:
        sonata_result: Result from sonata_analyze().
        runtime_values: Runtime tensor shapes/values to check against.
        verbose: If True, log each guard check.

    Returns:
        Tuple of GuardCheckResult, one per region with guards.
    """
    from .guard import GuardEvaluator, GUARD_SEVERITY_HARD

    evaluator = GuardEvaluator()
    results: list[GuardCheckResult] = []

    for region_id, status in sonata_result.region_statuses.items():
        # Collect guards from the score's shape assumptions
        if sonata_result.score is None:
            results.append(GuardCheckResult(region_id=region_id, guard_status="all_satisfied"))
            continue

        guards = list(sonata_result.score.shape_assumptions)
        if not guards:
            results.append(GuardCheckResult(region_id=region_id, guard_status="all_satisfied"))
            continue

        violated: list[str] = []
        details: list[GuardDetail] = []
        any_failed = False
        hard_failed = False

        for guard in guards:
            satisfied, action = evaluator.evaluate(guard, runtime_values)
            details.append(GuardDetail(
                symbol=guard.symbol,
                satisfied=satisfied,
                severity=str(guard.severity),
            ))
            if not satisfied:
                violated.append(guard.symbol)
                any_failed = True
                if guard.severity == GUARD_SEVERITY_HARD:
                    hard_failed = True

        if hard_failed:
            gs = "all_failed"
        elif any_failed:
            # v0.17 Phase 2 B2: STALE = only soft guards failed,
            # Score fingerprint still valid, only plan handle needs rebuild
            gs = "stale"
        else:
            gs = "all_satisfied"

        results.append(GuardCheckResult(
            region_id=region_id,
            guard_status=gs,
            violated_guards=tuple(violated),
            guard_details=tuple(details),
        ))

        if verbose and any_failed:
            _region_log.warning(
                "[GUARD] %s: %s — violated: %s",
                region_id, gs, violated,
            )

    return tuple(results)


def update_region_guard_status(
    plan_handle: Any,
    guard_results: tuple[GuardCheckResult, ...],
) -> dict[str, str]:
    """Update region_guard_status from runtime guard check results.

    v0.12 Phase 2 B2: region_guard_status update mechanism.

    Args:
        plan_handle: PlanHandle to update (frozen — returns new dict).
        guard_results: Results from check_guards_at_runtime.

    Returns:
        Updated region_guard_status dict.
    """
    from .plan_handle import GuardStatus

    status_map: dict[str, GuardStatus] = {}
    for gr in guard_results:
        try:
            status_map[gr.region_id] = GuardStatus(gr.guard_status)
        except ValueError:
            status_map[gr.region_id] = GuardStatus.ALL_FAILED

    return {k: v for k, v in status_map.items()}


def invalidate_on_guard_violation(
    guard_results: tuple[GuardCheckResult, ...],
    region_tree: Any,
    cache: Any,
    *,
    path_to_fingerprint: dict[str, str] | None = None,
) -> int:
    """Invalidate cache entries for regions with guard violations.

    v0.12 Phase 2 B1: Connects guard checking to actual cache invalidation.
    For each region with ALL_FAILED or PARTIAL_FAILED guard status,
    invalidates that region and all its descendants in the cache.

    Args:
        guard_results: Results from check_guards_at_runtime().
        region_tree: RegionTree for path resolution.
        cache: ScoreCache instance.
        path_to_fingerprint: Path → fingerprint mapping from store_region_tree.

    Returns:
        Total number of cache entries invalidated.
    """
    from .regions import invalidate_region_tree

    total_invalidated = 0
    for gr in guard_results:
        if gr.guard_status == "all_satisfied":
            continue

        # Find the matching node in the region tree
        target_node = None
        for node in region_tree.all_nodes:
            if f"region_{node.region.region_id}" == gr.region_id:
                target_node = node
                break

        if target_node is not None and path_to_fingerprint is not None:
            count = invalidate_region_tree(
                region_tree, target_node, cache,
                path_to_fingerprint=path_to_fingerprint,
            )
            total_invalidated += count

            _region_log.warning(
                "[INVALIDATE] %s: %s — invalidated %d cache entries (violated: %s)",
                gr.region_id, gr.guard_status, count, gr.violated_guards,
            )

    return total_invalidated
