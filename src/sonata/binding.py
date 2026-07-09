# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Sonata schedule binding layer — fill late-bound func_id and runtime_slot.

Takes a ``SonataScheduleContract`` produced by ``build_schedule()`` (where
``func_id`` and ``runtime_slot`` are ``None``) and binds them using real codegen
output::

    kernel_identity (stable logical callee name)
      → func_id (codegen-assigned via OrchestrationResult.func_name_to_id)

    arg_identity (stable logical tensor/scalar name)
      → runtime_slot (positional index in ChipStorageTaskArgs)

Both operations are pure data transformations on immutable dataclasses.
Failure to bind a value leaves it ``None`` with a ``FallbackReason``.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable

from .fallback import FallbackCode
from .score import FallbackReason
from .schedule import (
    ArgBinding,
    ScheduledRegion,
    ScheduledTask,
    SonataScheduleContract,
)

log = logging.getLogger("sonata.binding")


def _map_tasks(
    schedule: SonataScheduleContract,
    task_fn: Callable[[ScheduledTask], ScheduledTask],
) -> SonataScheduleContract:
    """Apply ``task_fn`` to every task in every static region.

    Dynamic regions are left unchanged. This is the shared building block
    for both ``bind_func_ids`` and ``bind_runtime_slots``.
    """
    new_regions: list[ScheduledRegion] = []
    for r in schedule.regions:
        if r.kind != "static" or not r.tasks:
            new_regions.append(r)
            continue
        new_tasks = tuple(task_fn(t) for t in r.tasks)
        new_regions.append(dataclasses.replace(r, tasks=new_tasks))
    return dataclasses.replace(schedule, regions=tuple(new_regions))


def bind_func_ids(
    schedule: SonataScheduleContract,
    func_name_to_id: dict[str, int],
    overrides: dict[str, int] | None = None,
) -> tuple[SonataScheduleContract, tuple[FallbackReason, ...]]:
    """Bind ``kernel_identity`` → ``func_id`` using codegen's func_name_to_id map.

    Args:
        schedule: Unbound schedule (``func_id`` is ``None``).
        func_name_to_id: Mapping from codegen: ``{kernel_name: func_id}``.
        overrides: Optional manual overrides ({kernel_name: func_id}),
            applied with higher priority than func_name_to_id.

    Returns:
        ``(bound_schedule, reasons)`` where ``reasons`` contains a
        ``FallbackReason`` for each task whose ``kernel_identity`` was not
        found in either ``func_name_to_id`` or ``overrides``.
    """
    merged: dict[str, int] = {}
    merged.update(func_name_to_id)
    if overrides:
        merged.update(overrides)  # overrides win

    reasons: list[FallbackReason] = []

    def mapper(task: ScheduledTask) -> ScheduledTask:
        if task.kernel_identity in merged:
            return dataclasses.replace(task, func_id=merged[task.kernel_identity])
        reasons.append(FallbackReason(
            code=FallbackCode.BINDING_FUNC_ID_NOT_FOUND,
            message=f"kernel_identity {task.kernel_identity!r} not found in func_name_to_id",
        ))
        return dataclasses.replace(task, func_id=None)  # force INVALID_KERNEL_ID in binary

    result = _map_tasks(schedule, mapper)
    return result, tuple(reasons)


def bind_runtime_slots(
    schedule: SonataScheduleContract,
    tensor_names: list[str],
    scalar_names: list[str],
) -> tuple[SonataScheduleContract, tuple[FallbackReason, ...]]:
    """Bind ``arg_identity`` → ``runtime_slot`` using positional ordering.

    Codegen's ``BuildTaskParams`` uses ``std::stable_partition`` to group
    tensor args before scalar args while preserving within-group declaration
    order. This function mirrors that ordering::

        tensor args → slots 0, 1, ..., N-1  (in tensor_names order)
        scalar args → slots N, N+1, ...      (in scalar_names order)

    Args:
        schedule: Schedule with ``arg_identity`` populated and
            ``runtime_slot`` as ``None``.
        tensor_names: List of tensor arg names in ``ChipStorageTaskArgs``
            order (all tensors, then all scalars).
        scalar_names: List of scalar arg names, also in declaration order.

    Returns:
        ``(bound_schedule, reasons)`` where ``reasons`` contains a
        ``FallbackReason`` for each arg whose ``arg_identity`` was not
        found in either ``tensor_names`` or ``scalar_names``.
    """
    tensor_slots = {name: i for i, name in enumerate(tensor_names)}
    scalar_offset = len(tensor_names)
    scalar_slots = {name: i for i, name in enumerate(scalar_names)}

    reasons: list[FallbackReason] = []

    def mapper(task: ScheduledTask) -> ScheduledTask:
        new_args: list[ArgBinding] = []
        for arg in task.args:
            if arg.arg_identity in tensor_slots:
                new_args.append(dataclasses.replace(
                    arg, runtime_slot=tensor_slots[arg.arg_identity],
                ))
            elif arg.arg_identity in scalar_slots:
                new_args.append(dataclasses.replace(
                    arg, runtime_slot=scalar_offset + scalar_slots[arg.arg_identity],
                ))
            else:
                reasons.append(FallbackReason(
                    code=FallbackCode.BINDING_MISSING_SLOT,
                    message=f"arg_identity {arg.arg_identity!r} not found in tensor or scalar names",
                ))
                new_args.append(arg)  # leave runtime_slot as None
        return dataclasses.replace(task, args=tuple(new_args))

    result = _map_tasks(schedule, mapper)
    return result, tuple(reasons)


__all__ = [
    "bind_func_ids",
    "bind_runtime_slots",
]
