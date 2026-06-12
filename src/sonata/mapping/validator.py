# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TMARB mapping validator — structural invariants for TMARB consumption.

Checks that a ``SonataScheduleContract`` can be losslessly mapped to TMARB's
runtime API without producing structurally invalid call sequences.
"""

from __future__ import annotations

from ..fallback import FallbackCode
from ..score import FallbackReason
from ..schedule import ScheduledRegion, ScopeMode, SonataScheduleContract


def validate_tmarb_mapping(
    schedule: SonataScheduleContract,
) -> tuple[FallbackReason, ...]:
    """Validate that ``schedule`` can be consumed by the TMARB mapping layer.

    Returns an empty tuple when the schedule is valid for TMARB consumption.
    Each ``FallbackReason`` describes a structural issue that would produce
    an invalid TMARB call sequence.
    """
    reasons: list[FallbackReason] = []

    _check_mixed_kernels(schedule.regions, reasons)
    _check_scope_nesting(schedule.regions, reasons)
    _check_unbound_func_ids(schedule.regions, reasons)

    return tuple(reasons)


def _check_mixed_kernels(
    regions: tuple[ScheduledRegion, ...],
    reasons: list[FallbackReason],
) -> None:
    for region in regions:
        if region.kind != "static":
            continue
        for task in region.tasks:
            if task.core_type == "mixed" and task.mixed_kernels is None:
                reasons.append(FallbackReason(
                    code=FallbackCode.SCHEDULE_BAD_NULLABLE,
                    message=f"region {region.region_id} task {task.task_id}: "
                            f"core_type='mixed' requires mixed_kernels",
                ))


def _check_scope_nesting(
    regions: tuple[ScheduledRegion, ...],
    reasons: list[FallbackReason],
) -> None:
    """TMARB forbids AUTO scope nested inside MANUAL scope.

    Scopes are sequential (not nested) — each region closes its scope
    before the next opens. Flag AUTO after MANUAL as suspicious.
    """
    prev_manual = False
    for region in regions:
        if region.kind != "static":
            prev_manual = False
            continue
        if region.scope_mode == ScopeMode.MANUAL:
            prev_manual = True
        elif prev_manual:
            reasons.append(FallbackReason(
                code=FallbackCode.SCHEDULE_DYNAMIC_DEP,
                message=f"region {region.region_id}: AUTO scope after MANUAL scope "
                        f"(AUTO inside MANUAL is forbidden by TMARB)",
            ))
            prev_manual = False


def _check_unbound_func_ids(
    regions: tuple[ScheduledRegion, ...],
    reasons: list[FallbackReason],
) -> None:
    for region in regions:
        if region.kind != "static":
            continue
        for task in region.tasks:
            if task.func_id is None:
                reasons.append(FallbackReason(
                    code=FallbackCode.BINDING_FUNC_ID_NOT_FOUND,
                    message=f"region {region.region_id} task {task.task_id} "
                            f"({task.kernel_identity}): func_id is None, "
                            f"cannot produce valid submit call",
                    severity="warning",
                ))


__all__ = [
    "validate_tmarb_mapping",
]
