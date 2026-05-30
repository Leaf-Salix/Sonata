# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dependency policy helpers for Sonata scores.

The policies operate only on the pure-Python :class:`Task` model. This keeps
Sonata's first dependency experiments decoupled from PyPTO's C++ IR bindings.
"""

from .fallback import FallbackCode
from .score import Dependency, Task

DEPENDENCY_POLICY_SEQUENTIAL_V0 = "sequential_v0"
DEPENDENCY_POLICY_DATAFLOW_V0 = "dataflow_v0"

_READ_DIRECTIONS = {"input", "inout"}
_WRITE_DIRECTIONS = {"output", "outputexisting", "inout"}
_IGNORED_DIRECTIONS = {"scalar", "nodep"}


def build_dependencies(
    tasks: tuple[Task, ...],
    *,
    policy: str = DEPENDENCY_POLICY_SEQUENTIAL_V0,
) -> tuple[Dependency, ...]:
    """Build task dependencies according to a named Sonata policy."""
    if policy == DEPENDENCY_POLICY_SEQUENTIAL_V0:
        return build_sequential_dependencies(tasks)
    if policy == DEPENDENCY_POLICY_DATAFLOW_V0:
        return build_dataflow_dependencies(tasks)
    raise ValueError(f"unsupported Sonata dependency policy: {policy}")


def build_sequential_dependencies(tasks: tuple[Task, ...]) -> tuple[Dependency, ...]:
    """Chain tasks in extraction order."""
    return tuple(
        Dependency(producer=tasks[index].task_id, consumer=tasks[index + 1].task_id)
        for index in range(len(tasks) - 1)
    )


def build_dataflow_dependencies(tasks: tuple[Task, ...]) -> tuple[Dependency, ...]:
    """Build conservative RAW/WAW/WAR edges from task args and directions.

    Every task must carry ``arg_directions``. If directions are unavailable,
    callers should keep using ``sequential_v0`` instead of this policy.
    """
    _require_complete_directions(tasks)

    edges: set[tuple[int, int]] = set()
    last_writer: dict[object, int] = {}
    readers_since_write: dict[object, set[int]] = {}

    for task in tasks:
        reads, writes = _read_write_args(task)

        for access_key in reads:
            writer = last_writer.get(access_key)
            if writer is not None and writer != task.task_id:
                edges.add((writer, task.task_id))
            readers_since_write.setdefault(access_key, set()).add(task.task_id)

        for access_key in writes:
            writer = last_writer.get(access_key)
            if writer is not None and writer != task.task_id:
                edges.add((writer, task.task_id))
            for reader in readers_since_write.get(access_key, set()):
                if reader != task.task_id:
                    edges.add((reader, task.task_id))
            last_writer[access_key] = task.task_id
            readers_since_write[access_key] = set()

    return tuple(Dependency(producer=producer, consumer=consumer) for producer, consumer in sorted(edges))


def supports_dataflow_dependencies(tasks: tuple[Task, ...]) -> FallbackCode | None:
    """Return None when all tasks carry enough direction data, or a FallbackCode
    explaining why dataflow dependencies cannot be built."""
    if not tasks:
        return None
    has_any = [bool(task.arg_directions) for task in tasks]
    has_complete = [h and len(task.arg_directions) == len(task.args) for h, task in zip(has_any, tasks)]
    if all(has_complete):
        return None
    if not any(has_any):
        return FallbackCode.DATAFLOW_DIRECTIONS_UNAVAILABLE
    return FallbackCode.DATAFLOW_DIRECTIONS_INCOMPLETE


def _require_complete_directions(tasks: tuple[Task, ...]) -> None:
    code = supports_dataflow_dependencies(tasks)
    if code is None:
        return
    missing = [str(task.task_id) for task in tasks if not task.arg_directions]
    mismatched = [
        str(task.task_id)
        for task in tasks
        if task.arg_directions and len(task.arg_directions) != len(task.args)
    ]
    details = []
    if missing:
        details.append(f"missing directions for tasks: {', '.join(missing)}")
    if mismatched:
        details.append(f"direction/arg size mismatch for tasks: {', '.join(mismatched)}")
    raise ValueError(f"dataflow_v0 requires complete task arg_directions ({'; '.join(details)})")


def _read_write_args(task: Task) -> tuple[set[object], set[object]]:
    reads: set[object] = set()
    writes: set[object] = set()
    for arg, direction, storage_key in zip(
        task.args,
        task.arg_directions,
        _storage_keys(task),
        strict=True,
    ):
        normalized = _normalize_direction(direction)
        if normalized in _IGNORED_DIRECTIONS:
            continue
        access_key = storage_key if storage_key is not None else arg
        if normalized in _READ_DIRECTIONS:
            reads.add(access_key)
        if normalized in _WRITE_DIRECTIONS:
            writes.add(access_key)
    return reads, writes


def _storage_keys(task: Task) -> tuple[object | None, ...]:
    if task.arg_storage_keys:
        return task.arg_storage_keys
    return (None,) * len(task.args)


def _normalize_direction(direction: str) -> str:
    return "".join(ch for ch in str(direction).lower() if ch.isalnum())


__all__ = [
    "DEPENDENCY_POLICY_DATAFLOW_V0",
    "DEPENDENCY_POLICY_SEQUENTIAL_V0",
    "build_dataflow_dependencies",
    "build_dependencies",
    "build_sequential_dependencies",
    "supports_dataflow_dependencies",
]
