# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Buffer liveness analysis for Sonata v0.3 storage model.

Computes the task-level lifetime interval [birth, death] for each storage
key, where birth is the first task that writes the buffer and death is the
last task that reads it. This is a graph-level analysis, not kernel-level.

Touching lifetimes (death of A == birth of B) can share memory,
matching PyPTO MemoryReuse semantics: within a single statement,
inputs are consumed before outputs are produced.
"""

from dataclasses import dataclass

from .directions import READ_DIRECTIONS, WRITE_DIRECTIONS, normalize_direction
from .score import Task


@dataclass(frozen=True)
class BufferLifetime:
    """Lifetime interval for one storage buffer across the task sequence."""

    storage_key: str
    birth: int
    death: int

    def overlaps(self, other: "BufferLifetime") -> bool:
        """Return whether two lifetimes overlap (both alive at the same time).

        Touching lifetimes (self.death == other.birth) do NOT overlap,
        matching PyPTO MemoryReuse: inputs consumed before outputs produced.
        """
        return self.birth < other.death and other.birth < self.death


@dataclass(frozen=True)
class StorageConflict:
    """Two buffers that are simultaneously alive."""

    key_a: str
    key_b: str


def compute_lifetimes(tasks: tuple[Task, ...]) -> tuple[BufferLifetime, ...]:
    """Compute buffer lifetimes from task args, directions, storage keys, and outputs.

    Birth = first task_id that writes the buffer.
    Death = last task_id that reads or writes the buffer.

    v0.11 Phase 3 B1: When ``Task.outputs`` is non-empty, each output
    buffer is tracked as an explicit write (more accurate than inferring
    from arg_directions alone).
    """
    births: dict[str, int] = {}
    deaths: dict[str, int] = {}

    for task in tasks:
        storage_keys = task.arg_storage_keys if task.arg_storage_keys else ()
        directions = task.arg_directions if task.arg_directions else ()

        for index, (arg, direction) in enumerate(zip(task.args, directions)):
            normalized = normalize_direction(direction)
            sk = storage_keys[index] if index < len(storage_keys) else None
            if sk is None:
                continue

            if normalized in WRITE_DIRECTIONS:
                if sk not in births:
                    births[sk] = task.task_id
                deaths[sk] = task.task_id

            if normalized in READ_DIRECTIONS:
                if sk not in births:
                    births[sk] = task.task_id
                deaths[sk] = max(deaths.get(sk, task.task_id), task.task_id)

        # v0.11: explicit outputs tracked as writes
        for out_key in task.outputs:
            if out_key not in births:
                births[out_key] = task.task_id
            deaths[out_key] = max(deaths.get(out_key, task.task_id), task.task_id)

        # v0.21 Phase 4 A2: Typed output_defs for precise liveness
        if task.output_defs is not None:
            for out_def in task.output_defs:
                key = out_def.buffer_id
                if key not in births:
                    births[key] = task.task_id
                deaths[key] = max(deaths.get(key, task.task_id), task.task_id)

    result: list[BufferLifetime] = []
    for key in sorted(births.keys()):
        result.append(BufferLifetime(
            storage_key=key,
            birth=births[key],
            death=deaths.get(key, births[key]),
        ))
    return tuple(result)


def find_conflicts(
    lifetimes: tuple[BufferLifetime, ...],
) -> tuple[StorageConflict, ...]:
    """Find pairs of buffers with overlapping lifetimes."""
    conflicts: list[StorageConflict] = []
    sorted_lifetimes = sorted(lifetimes, key=lambda lt: lt.birth)
    for i, lt_a in enumerate(sorted_lifetimes):
        for lt_b in sorted_lifetimes[i + 1:]:
            if lt_b.birth > lt_a.death:
                break
            if lt_a.overlaps(lt_b):
                conflicts.append(StorageConflict(
                    key_a=lt_a.storage_key,
                    key_b=lt_b.storage_key,
                ))
    return tuple(conflicts)


__all__ = [
    "BufferLifetime",
    "StorageConflict",
    "compute_lifetimes",
    "find_conflicts",
]
