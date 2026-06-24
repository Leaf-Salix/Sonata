# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TMARB call trace generation — deterministic API call sequence from a schedule contract.

Produces a ``list[TMARBCallTraceEntry]`` showing exactly what TMARB API calls
a ``SonataScheduleContract`` maps to. The trace is deterministic, diffable,
and can be compared against traces extracted from real codegen output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..schedule import (
    ArgBinding,
    ArgDirection,
    ScheduleDep,
    ScheduledRegion,
    ScheduledTask,
    SonataScheduleContract,
)


@dataclass(frozen=True)
class TMARBCallTraceEntry:
    """One TMARB API call in a deterministic trace.

    Each entry maps to a single TMARB operation: ``add_input``,
    ``rt_submit_aic_task``, ``PTO2_SCOPE``, etc.
    """
    phase: str          # "entry_setup" | "alloc" | "region" | "task"
    api: str            # "from_tensor_arg" | "add_input" | "rt_submit_aic_task" | ...
    args: dict[str, Any] = field(default_factory=dict)
    result: str | list[str] | None = None
    region_id: str | None = None
    region_kind: str | None = None
    task_id: int | None = None
    task_id_ref: str | None = None


def generate_trace(schedule: SonataScheduleContract) -> list[TMARBCallTraceEntry]:
    """Generate a deterministic TMARB call trace from a schedule contract.

    The trace shows the exact sequence of TMARB API calls a runtime
    consumer would make. Same schedule → same trace (no randomness).
    """
    trace: list[TMARBCallTraceEntry] = []
    task_to_ref: dict[int, str] = {}  # task_id → "t<N>.task_id()"

    for region in schedule.regions:
        _emit_region(region, trace, task_to_ref)

    return trace


def trace_to_json(trace: list[TMARBCallTraceEntry], indent: int = 2) -> str:
    """Serialize a trace to deterministic JSON."""
    data = [_entry_to_dict(e) for e in trace]
    return json.dumps(data, indent=indent, sort_keys=True)


def _entry_to_dict(e: TMARBCallTraceEntry) -> dict[str, Any]:
    d: dict[str, Any] = {"phase": e.phase, "api": e.api}
    if e.args:
        d["args"] = dict(e.args)
    if e.result is not None:
        d["result"] = e.result
    if e.region_id is not None:
        d["region_id"] = e.region_id
    if e.region_kind is not None:
        d["region_kind"] = e.region_kind
    if e.task_id is not None:
        d["task_id"] = e.task_id
    if e.task_id_ref is not None:
        d["task_id_ref"] = e.task_id_ref
    return d


def _emit_region(
    region: ScheduledRegion,
    trace: list[TMARBCallTraceEntry],
    task_to_ref: dict[int, str],
) -> None:
    if region.kind == "dynamic":
        trace.append(TMARBCallTraceEntry(
            phase="region",
            region_id=region.region_id,
            region_kind="dynamic",
            api="PTO2_SCOPE",
            args={"mode": "auto"},
        ))
        return

    scope_mode = region.scope_mode.value if region.scope_mode else "auto"
    trace.append(TMARBCallTraceEntry(
        phase="region",
        region_id=region.region_id,
        region_kind="static",
        api="PTO2_SCOPE",
        args={"mode": scope_mode},
    ))

    for task in region.tasks:
        _emit_task(task, trace, task_to_ref)

    for dep in region.deps:
        _emit_dep(dep, trace)


def _emit_task(
    task: ScheduledTask,
    trace: list[TMARBCallTraceEntry],
    task_to_ref: dict[int, str],
) -> None:
    task_ref = f"t{task.task_id}"
    task_to_ref[task.task_id] = f"{task_ref}.task_id()"

    for arg in task.args:
        _emit_arg(arg, task, trace)

    if task.mixed_kernels is not None:
        trace.append(TMARBCallTraceEntry(
            phase="task", region_kind="static",
            task_id=task.task_id,
            api="MixedKernels",
            args={
                "aic_func_id": task.mixed_kernels.aic_func_id,
                "aiv_func_id": task.mixed_kernels.aiv_func_id,
                "dual_aiv_func_id": task.mixed_kernels.dual_aiv_func_id,
            },
        ))
        submit_api = "rt_submit_task"
    elif task.core_type == "aic":
        submit_api = "rt_submit_aic_task"
    elif task.core_type == "aiv":
        submit_api = "rt_submit_aiv_task"
    else:
        submit_api = "rt_submit_aic_task"

    tensor_count = sum(1 for a in task.args if a.direction != ArgDirection.SCALAR)
    scalar_count = sum(1 for a in task.args if a.direction == ArgDirection.SCALAR)
    trace.append(TMARBCallTraceEntry(
        phase="task", region_kind="static",
        task_id=task.task_id,
        api=submit_api,
        args={"func_id": task.func_id, "tensor_count": tensor_count, "scalar_count": scalar_count},
        result=f"TaskOutputTensors {task_ref}",
        task_id_ref=task_to_ref[task.task_id],
    ))


def _emit_arg(arg: ArgBinding, task: ScheduledTask, trace: list[TMARBCallTraceEntry]) -> None:
    dir_to_api = {
        ArgDirection.INPUT: "add_input",
        ArgDirection.OUTPUT: "add_output",
        ArgDirection.INOUT: "add_inout",
        ArgDirection.OUTPUT_EXISTING: "add_output",
        ArgDirection.NO_DEP: "add_no_dep",
        ArgDirection.SCALAR: "add_scalar",
    }
    api_name = dir_to_api.get(arg.direction, "add_input")

    a: dict[str, Any] = {"arg_identity": arg.arg_identity}
    if arg.runtime_slot is not None:
        a["runtime_slot"] = arg.runtime_slot

    trace.append(TMARBCallTraceEntry(
        phase="task", region_kind="static",
        task_id=task.task_id,
        api=api_name,
        args=a,
    ))


def _emit_dep(dep: ScheduleDep, trace: list[TMARBCallTraceEntry]) -> None:
    trace.append(TMARBCallTraceEntry(
        phase="dep",
        api="set_dependencies",
        args={"producer": dep.producer, "consumer": dep.consumer, "kind": dep.kind},
    ))


__all__ = [
    "TMARBCallTraceEntry",
    "generate_trace",
    "trace_to_json",
]
