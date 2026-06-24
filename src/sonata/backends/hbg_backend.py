# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""HBG Schedule Backend — consume SonataScheduleContract via host_build_graph.

This is the **correctness runway** for v0.23. It validates that a schedule
contract maps correctly to HBG task/edge topology. It does NOT validate
TMARB ``submit_task`` / ``set_dependencies`` / TensorMap / scope semantics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..schedule import SonataScheduleContract
from ..fallback import FallbackCode
from ..schedule_validator import validate_schedule
from ..runtime_adapter import HostBuildGraphEdge, HostBuildGraphPlan, HostBuildGraphTask
from ..score import FallbackReason

log = logging.getLogger("sonata.hbg_backend")


@dataclass(frozen=True)
class HBGScheduleResult:
    """Result of consuming a schedule contract through HBG."""

    success: bool
    plan: HostBuildGraphPlan | None = None
    reasons: tuple[FallbackReason, ...] = ()
    tasks: int = 0
    edges: int = 0

    @classmethod
    def accept(cls, plan: HostBuildGraphPlan) -> "HBGScheduleResult":
        return cls(
            success=True,
            plan=plan,
            tasks=plan.task_count(),
            edges=plan.edge_count(),
        )

    @classmethod
    def reject(cls, *reasons: FallbackReason) -> "HBGScheduleResult":
        return cls(success=False, reasons=reasons)


class HBGScheduleBackend:
    """HBG consumer for SonataScheduleContract — topology correctness validation.

    Limitations (must be communicated in docstrings and test output):
    - Only validates topology correctness.
    - Does NOT validate TMARB submit_task / set_dependencies / TensorMap / scope semantics.
    - func_id values from the schedule are passed through unchanged;
      actual func_id binding is a v0.24 concern.
    """

    def consume(self, schedule: SonataScheduleContract) -> HBGScheduleResult:
        errors = validate_schedule(schedule)
        if errors:
            log.info("[HBG] schedule validation failed: %s", errors)
            return HBGScheduleResult.reject(*errors)

        tasks: list[HostBuildGraphTask] = []
        edges: list[HostBuildGraphEdge] = []

        for region in schedule.regions:
            if region.kind != "static":
                continue

            for sched_task in region.tasks:
                hbg_task = HostBuildGraphTask(
                    task_id=sched_task.task_id,
                    func_id=sched_task.func_id if sched_task.func_id is not None else sched_task.task_id,
                    core_type=sched_task.core_type,
                    name=sched_task.name or sched_task.kernel_identity,
                )
                tasks.append(hbg_task)

            for dep in region.deps:
                edges.append(HostBuildGraphEdge(
                    producer=dep.producer,
                    consumer=dep.consumer,
                ))

        plan = HostBuildGraphPlan(
            tasks=tuple(tasks),
            edges=tuple(edges),
            metadata={
                "source": "sonata_schedule_contract",
                "fingerprint": schedule.fingerprint,
                "schema_version": schedule.schema_version,
            },
        )

        # Validate using existing adapter validation to catch self-edges etc.
        validation_errors = _validate_plan_topology(tasks, edges)
        if validation_errors:
            return HBGScheduleResult.reject(*validation_errors)

        return HBGScheduleResult.accept(plan)


def _validate_plan_topology(
    tasks: list[HostBuildGraphTask],
    edges: list[HostBuildGraphEdge],
) -> tuple[FallbackReason, ...]:
    reasons: list[FallbackReason] = []
    task_ids = {t.task_id for t in tasks}

    for edge in edges:
        if edge.producer == edge.consumer:
            reasons.append(FallbackReason(
                code=FallbackCode.SCHEDULE_SELF_EDGE,
                message=f"self-edge detected: task {edge.producer} depends on itself",
            ))
        if edge.producer not in task_ids:
            reasons.append(FallbackReason(
                code=FallbackCode.SCHEDULE_INVALID_EDGE,
                message=f"producer {edge.producer} not in task table",
            ))
        if edge.consumer not in task_ids:
            reasons.append(FallbackReason(
                code=FallbackCode.SCHEDULE_INVALID_EDGE,
                message=f"consumer {edge.consumer} not in task table",
            ))

    return tuple(reasons)


__all__ = [
    "HBGScheduleBackend",
    "HBGScheduleResult",
]
