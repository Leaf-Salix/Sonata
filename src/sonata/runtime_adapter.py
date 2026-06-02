# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""HostBuildGraph runtime adapter for Sonata v0.2.

Generates or validates host-build-graph-shaped task tables and explicit edge
contracts from Score + PlanHandle. Validation failures produce structured
FallbackReason entries so the caller can fall back to the original PyPTO
compile/runtime path.
"""

from dataclasses import dataclass, field
from typing import Any

from .directions import MEMORY_DIRECTIONS, normalize_direction
from .fallback import FallbackCode
from .plan_handle import (
    FuncRegistry,
    PlanHandle,
    RuntimeArgBinding,
    RUNTIME_CONTRACT_VERSION,
)
from .score import (
    Dependency,
    EligibilityResult,
    FallbackReason,
    Score,
    Task,
)
from .serialization import score_fingerprint


@dataclass(frozen=True)
class HostBuildGraphTask:
    """One task in a host-build-graph runtime plan."""

    task_id: int
    func_id: int
    core_type: str
    runtime_args: tuple[Any, ...] = ()
    name: str | None = None


@dataclass(frozen=True)
class HostBuildGraphEdge:
    """One explicit dependency edge in a host-build-graph runtime plan."""

    producer: int
    consumer: int


@dataclass(frozen=True)
class HostBuildGraphPlan:
    """Structured output for the host_build_graph runtime.

    Contains the task table, edge table, and audit metadata needed by
    the host-side graph builder.
    """

    tasks: tuple[HostBuildGraphTask, ...] = ()
    edges: tuple[HostBuildGraphEdge, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def task_count(self) -> int:
        return len(self.tasks)

    def edge_count(self) -> int:
        return len(self.edges)


@dataclass(frozen=True)
class RuntimeAdapterResult:
    """Result of a runtime adapter generation or validation."""

    success: bool
    plan: HostBuildGraphPlan | None = None
    reasons: tuple[FallbackReason, ...] = ()

    @classmethod
    def accept(cls, plan: HostBuildGraphPlan) -> "RuntimeAdapterResult":
        return cls(success=True, plan=plan)

    @classmethod
    def reject(cls, *reasons: FallbackReason) -> "RuntimeAdapterResult":
        return cls(success=False, reasons=reasons)


class HostBuildGraphRuntimeAdapter:
    """Generate and validate host-build-graph runtime plans.

    Translates Score + PlanHandle into a HostBuildGraphPlan, performing
    structural validation to catch mismatches before they reach the
    runtime layer.
    """

    def generate(
        self,
        score: Score,
        plan_handle: PlanHandle,
    ) -> RuntimeAdapterResult:
        """Generate a HostBuildGraphPlan from Score and PlanHandle."""
        reasons = self.validate(score, plan_handle)
        if reasons:
            return RuntimeAdapterResult.reject(*reasons)

        binding_map = _build_binding_map(plan_handle.arg_bindings)
        tasks = _build_tasks(score.tasks, plan_handle.func_registry, binding_map)
        edges = _build_edges(score.dependencies)
        metadata = _build_metadata(score, plan_handle)

        return RuntimeAdapterResult.accept(
            HostBuildGraphPlan(tasks=tasks, edges=edges, metadata=metadata)
        )

    def generate_region_aware(
        self,
        score: Score,
        plan_handle: PlanHandle,
        *,
        region_statuses: dict[str, str] | None = None,
    ) -> RuntimeAdapterResult:
        """Generate a region-aware HostBuildGraphPlan.

        v0.11 Phase 5 D: For mixed graphs, static regions produce normal
        tasks while dynamic regions are annotated as fallback in the plan
        metadata.

        Args:
            score: Score with tasks and dependencies.
            plan_handle: PlanHandle with bindings and runtime info.
            region_statuses: Optional mapping of region_id → status
                ("static", "dynamic", "mixed"). When provided, dynamic
                regions are flagged in the plan metadata.

        Returns:
            RuntimeAdapterResult with region-aware metadata.
        """
        base = self.generate(score, plan_handle)
        if not base.success:
            return base

        plan = base.plan
        if plan is None:
            return base

        # Merge region info into plan metadata
        extra_meta: dict[str, Any] = {}
        if region_statuses:
            dynamic_regions = [k for k, v in region_statuses.items() if v == "dynamic"]
            mixed_regions = [k for k, v in region_statuses.items() if v == "mixed"]
            extra_meta["region_statuses"] = region_statuses
            extra_meta["dynamic_region_count"] = len(dynamic_regions)
            extra_meta["mixed_region_count"] = len(mixed_regions)
            extra_meta["static_region_count"] = sum(
                1 for v in region_statuses.values() if v == "static"
            )

        # Include plan_handle region_guard_status
        if plan_handle.region_guard_status:
            extra_meta["region_guard_status"] = {
                k: v.value for k, v in plan_handle.region_guard_status.items()
            }

        if extra_meta:
            merged_meta = {**plan.metadata, **extra_meta}
            plan = HostBuildGraphPlan(
                tasks=plan.tasks, edges=plan.edges, metadata=merged_meta
            )

        return RuntimeAdapterResult.accept(plan)

    def validate(
        self,
        score: Score,
        plan_handle: PlanHandle,
    ) -> tuple[FallbackReason, ...]:
        """Validate Score/PlanHandle structural consistency."""
        reasons: list[FallbackReason] = []

        fp = score_fingerprint(score)
        if fp != plan_handle.score_fingerprint:
            reasons.append(
                _reason(
                    FallbackCode.RUNTIME_ADAPTER_FINGERPRINT_MISMATCH,
                    f"PlanHandle fingerprint {plan_handle.score_fingerprint!r} "
                    f"does not match Score fingerprint {fp!r}",
                )
            )

        if plan_handle.runtime_contract_version != RUNTIME_CONTRACT_VERSION:
            reasons.append(
                _reason(
                    FallbackCode.RUNTIME_ADAPTER_CONTRACT_VERSION_MISMATCH,
                    f"PlanHandle runtime contract version "
                    f"{plan_handle.runtime_contract_version} "
                    f"does not match adapter version {RUNTIME_CONTRACT_VERSION}",
                )
            )

        task_ids = {task.task_id for task in score.tasks}
        registry_names = plan_handle.func_registry.names()

        for task in score.tasks:
            if task.name is not None and task.name not in registry_names:
                reasons.append(
                    _reason(
                        FallbackCode.RUNTIME_ADAPTER_FUNC_NOT_REGISTERED,
                        f"task {task.task_id} callee {task.name!r} "
                        f"not found in FuncRegistry",
                    )
                )

        for entry in plan_handle.func_registry.entries:
            has_ref = any(task.name == entry.name for task in score.tasks)
            if not has_ref:
                reasons.append(
                    _reason(
                        FallbackCode.RUNTIME_ADAPTER_FUNC_UNREFERENCED,
                        f"FuncRegistry entry {entry.name!r} "
                        f"not referenced by any Score task",
                    )
                )

        binding_map = _build_binding_map(plan_handle.arg_bindings)
        for task in score.tasks:
            missing = _check_arg_bindings(task, binding_map)
            for msg in missing:
                reasons.append(
                    _reason(FallbackCode.RUNTIME_ADAPTER_BINDING_INCOMPLETE, msg)
                )

        for dep in score.dependencies:
            if dep.producer not in task_ids:
                reasons.append(
                    _reason(
                        FallbackCode.RUNTIME_ADAPTER_INVALID_EDGE,
                        f"dependency producer {dep.producer} "
                        f"not in task table",
                    )
                )
            if dep.consumer not in task_ids:
                reasons.append(
                    _reason(
                        FallbackCode.RUNTIME_ADAPTER_INVALID_EDGE,
                        f"dependency consumer {dep.consumer} "
                        f"not in task table",
                    )
                )

        return tuple(reasons)


def _build_binding_map(
    bindings: tuple[RuntimeArgBinding, ...],
) -> dict[tuple[int, int], RuntimeArgBinding]:
    return {(b.task_id, b.arg_index): b for b in bindings}


def _check_arg_bindings(
    task: Task,
    binding_map: dict[tuple[int, int], RuntimeArgBinding],
) -> list[str]:
    missing: list[str] = []
    directions = task.arg_directions if task.arg_directions else ()
    storage_keys = task.arg_storage_keys if task.arg_storage_keys else ()
    for index, (arg, direction) in enumerate(zip(task.args, directions)):
        normalized = normalize_direction(direction)
        if normalized not in MEMORY_DIRECTIONS:
            continue
        key = (task.task_id, index)
        if key not in binding_map:
            sk = storage_keys[index] if index < len(storage_keys) else None
            missing.append(
                f"task {task.task_id} arg {index} ({arg}, storage={sk}) "
                f"has no RuntimeArgBinding"
            )
    return missing


def _build_tasks(
    tasks: tuple[Task, ...],
    registry: FuncRegistry,
    binding_map: dict[tuple[int, int], RuntimeArgBinding],
) -> tuple[HostBuildGraphTask, ...]:
    result: list[HostBuildGraphTask] = []
    for task in tasks:
        runtime_args = _resolve_runtime_args(task, binding_map)
        func_id = task.func_id
        entry = registry.by_name(task.name) if task.name else None
        if entry is not None and entry.runtime_func_id is not None:
            func_id = entry.runtime_func_id
        result.append(
            HostBuildGraphTask(
                task_id=task.task_id,
                func_id=func_id,
                core_type=task.core_type,
                runtime_args=runtime_args,
                name=task.name,
            )
        )
    return tuple(result)


def _resolve_runtime_args(
    task: Task,
    binding_map: dict[tuple[int, int], RuntimeArgBinding],
) -> tuple[Any, ...]:
    args: list[Any] = []
    for index in range(len(task.args)):
        binding = binding_map.get((task.task_id, index))
        if binding is not None:
            args.append(binding.runtime_handle)
        else:
            args.append(None)
    return tuple(args)


def _build_edges(
    dependencies: tuple[Dependency, ...],
) -> tuple[HostBuildGraphEdge, ...]:
    return tuple(
        HostBuildGraphEdge(producer=dep.producer, consumer=dep.consumer)
        for dep in dependencies
    )


def _build_metadata(
    score: Score,
    plan_handle: PlanHandle,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "score_fingerprint": plan_handle.score_fingerprint,
        "source_adapter": plan_handle.source_adapter,
        "runtime_contract_version": plan_handle.runtime_contract_version,
        "runtime_target": plan_handle.runtime_target.runtime,
        "schema_version": plan_handle.schema_version,
    }
    return meta


def _reason(code: FallbackCode, message: str) -> FallbackReason:
    return FallbackReason(code=code.value, message=message, severity="error")


__all__ = [
    "HostBuildGraphEdge",
    "HostBuildGraphPlan",
    "HostBuildGraphRuntimeAdapter",
    "HostBuildGraphTask",
    "RuntimeAdapterResult",
]
