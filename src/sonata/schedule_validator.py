# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Sonata schedule validator — invariant checks before backend consumption."""

from __future__ import annotations

from .fallback import FallbackCode
from .score import FallbackReason
from .schedule import RegionBoundary, ScheduledRegion, SonataScheduleContract


def validate_schedule(
    schedule: SonataScheduleContract,
    source_fingerprint: str | None = None,
) -> tuple[FallbackReason, ...]:
    reasons: list[FallbackReason] = []

    _check_dep_endpoints(schedule.regions, reasons)
    _check_region_id_uniqueness(schedule.regions, reasons)
    _check_acyclicity(schedule.regions, reasons)
    _check_dynamic_no_deps(schedule.regions, reasons)
    _check_dependency_closure(schedule.regions, reasons)
    _check_boundary_consistency(schedule.regions, schedule.boundaries, reasons)
    _check_nullable_fields(schedule.regions, reasons)
    _check_replay_neutral(schedule, reasons)

    if source_fingerprint is not None and schedule.fingerprint != source_fingerprint:
        reasons.append(FallbackReason(
            code=FallbackCode.SCHEDULE_FP_MISMATCH,
            message=f"schedule fingerprint {schedule.fingerprint!r} != source {source_fingerprint!r}",
        ))

    return tuple(reasons)


def _check_dep_endpoints(
    regions: tuple[ScheduledRegion, ...],
    reasons: list[FallbackReason],
) -> None:
    for region in regions:
        if region.kind != "static":
            continue
        task_ids = {t.task_id for t in region.tasks}
        for dep in region.deps:
            if dep.producer not in task_ids:
                reasons.append(FallbackReason(
                    code=FallbackCode.SCHEDULE_INVALID_DEP,
                    message=f"region {region.region_id}: dep producer {dep.producer} not in task list",
                ))
            if dep.consumer not in task_ids:
                reasons.append(FallbackReason(
                    code=FallbackCode.SCHEDULE_INVALID_DEP,
                    message=f"region {region.region_id}: dep consumer {dep.consumer} not in task list",
                ))


def _check_region_id_uniqueness(
    regions: tuple[ScheduledRegion, ...],
    reasons: list[FallbackReason],
) -> None:
    seen: set[str] = set()
    for region in regions:
        if region.region_id in seen:
            reasons.append(FallbackReason(
                code=FallbackCode.SCHEDULE_DUPLICATE_REGION,
                message=f"duplicate region_id: {region.region_id!r}",
            ))
        seen.add(region.region_id)


def _check_acyclicity(
    regions: tuple[ScheduledRegion, ...],
    reasons: list[FallbackReason],
) -> None:
    """Kahn's algorithm — detect cycles in static region dependency graphs."""
    for region in regions:
        if region.kind != "static" or len(region.tasks) < 2:
            continue
        in_degree = {t.task_id: 0 for t in region.tasks}
        adj: dict[int, list[int]] = {t.task_id: [] for t in region.tasks}
        for dep in region.deps:
            adj.setdefault(dep.producer, []).append(dep.consumer)
            in_degree[dep.consumer] = in_degree.get(dep.consumer, 0) + 1
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for succ in adj.get(node, []):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        if visited != len(in_degree):
            reasons.append(FallbackReason(
                code=FallbackCode.SCHEDULE_CYCLE,
                message=f"region {region.region_id}: cycle detected in dependency graph",
            ))


def _check_dynamic_no_deps(
    regions: tuple[ScheduledRegion, ...],
    reasons: list[FallbackReason],
) -> None:
    """Verify no dep producer/consumer references a task outside any static region."""
    static_task_ids: set[int] = set()
    for r in regions:
        if r.kind == "static":
            static_task_ids.update(t.task_id for t in r.tasks)
    if not static_task_ids:
        return
    for r in regions:
        if r.kind != "static":
            continue
        for dep in r.deps:
            if dep.producer not in static_task_ids:
                reasons.append(FallbackReason(
                    code=FallbackCode.SCHEDULE_DYNAMIC_DEP,
                    message=f"region {r.region_id}: dep producer {dep.producer} not in any static region",
                ))
            if dep.consumer not in static_task_ids:
                reasons.append(FallbackReason(
                    code=FallbackCode.SCHEDULE_DYNAMIC_DEP,
                    message=f"region {r.region_id}: dep consumer {dep.consumer} not in any static region",
                ))


def _check_dependency_closure(
    regions: tuple[ScheduledRegion, ...],
    reasons: list[FallbackReason],
) -> None:
    """Warn about orphan tasks that have no predecessor or successor."""
    for region in regions:
        if region.kind != "static" or len(region.tasks) <= 1:
            continue
        task_ids = {t.task_id for t in region.tasks}
        has_predecessor: set[int] = set()
        has_successor: set[int] = set()
        for dep in region.deps:
            has_predecessor.add(dep.consumer)
            has_successor.add(dep.producer)
        orphans = task_ids - has_predecessor - has_successor
        for tid in sorted(orphans):
            reasons.append(FallbackReason(
                code=FallbackCode.SCHEDULE_ORPHAN,
                message=f"region {region.region_id}: task {tid} has no incoming or outgoing edges",
            ))


def _check_boundary_consistency(
    regions: tuple[ScheduledRegion, ...],
    boundaries: tuple[RegionBoundary, ...],
    reasons: list[FallbackReason],
) -> None:
    region_ids = {r.region_id for r in regions}
    for b in boundaries:
        if b.from_region not in region_ids:
            reasons.append(FallbackReason(
                code=FallbackCode.SCHEDULE_BAD_BOUNDARY,
                message=f"boundary from_region {b.from_region!r} not in regions",
            ))
        if b.to_region not in region_ids:
            reasons.append(FallbackReason(
                code=FallbackCode.SCHEDULE_BAD_BOUNDARY,
                message=f"boundary to_region {b.to_region!r} not in regions",
            ))

        src_region = next((r for r in regions if r.region_id == b.from_region), None)
        if src_region and src_region.tasks and b.tensors:
            src_outputs: set[str] = set()
            for t in src_region.tasks:
                src_outputs.update(t.outputs)
            for tensor in b.tensors:
                if tensor not in src_outputs:
                    reasons.append(FallbackReason(
                        code=FallbackCode.SCHEDULE_BAD_BOUNDARY,
                        message=f"boundary tensor {tensor!r} not produced by source region {b.from_region!r}",
                    ))

        dst_region = next((r for r in regions if r.region_id == b.to_region), None)
        if dst_region and dst_region.tasks and b.tensors:
            dst_consumed: set[str] = set()
            for t in dst_region.tasks:
                for a in t.args:
                    dst_consumed.add(a.arg_identity)
            for tensor in b.tensors:
                if tensor not in dst_consumed:
                    reasons.append(FallbackReason(
                        code=FallbackCode.SCHEDULE_BAD_BOUNDARY,
                        message=f"boundary tensor {tensor!r} not consumed by target region {b.to_region!r}",
                    ))


def _check_nullable_fields(
    regions: tuple[ScheduledRegion, ...],
    reasons: list[FallbackReason],
) -> None:
    for region in regions:
        for task in region.tasks:
            if task.func_id is not None and (not isinstance(task.func_id, int) or task.func_id < 0):
                reasons.append(FallbackReason(
                    code=FallbackCode.SCHEDULE_BAD_NULLABLE,
                    message=f"region {region.region_id} task {task.task_id}: func_id must be None or non-negative int, got {task.func_id!r}",
                ))
            for arg in task.args:
                if arg.runtime_slot is not None and (not isinstance(arg.runtime_slot, int) or arg.runtime_slot < 0):
                    reasons.append(FallbackReason(
                        code=FallbackCode.SCHEDULE_BAD_NULLABLE,
                        message=f"region {region.region_id} task {task.task_id} arg {arg.arg_identity!r}: runtime_slot must be None or non-negative int, got {arg.runtime_slot!r}",
                    ))


def _check_replay_neutral(
    schedule: SonataScheduleContract,
    reasons: list[FallbackReason],
) -> None:
    d = schedule.to_dict()
    prohibited = {"orchestration_mode", "replay_strategy", "execution_mode"}
    found = prohibited & set(d.keys())
    if found:
        reasons.append(FallbackReason(
            code=FallbackCode.SCHEDULE_REPLAY_SPECIFIC,
            message=f"schedule contains replay-specific fields: {sorted(found)}",
        ))
    for region in d.get("regions", []):
        if isinstance(region, dict):
            region_found = prohibited & set(region.keys())
            if region_found:
                reasons.append(FallbackReason(
                    code=FallbackCode.SCHEDULE_REPLAY_SPECIFIC,
                    message=f"region {region.get('region_id', '?')} contains replay-specific fields: {sorted(region_found)}",
                ))


__all__ = [
    "validate_schedule",
]
