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

from .directions import IGNORED_DIRECTIONS, READ_DIRECTIONS, WRITE_DIRECTIONS, normalize_direction
from .fallback import FallbackCode
from .score import Dependency, DependencyKind, Task

DEPENDENCY_POLICY_SEQUENTIAL_V0 = "sequential_v0"
DEPENDENCY_POLICY_DATAFLOW_V0 = "dataflow_v0"

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


def build_ordering_dependencies(
    tasks: tuple[Task, ...],
    *,
    side_effect_tasks: frozenset[int] | None = None,
) -> tuple[Dependency, ...]:
    """Build pure ordering constraints between tasks.

    If ``side_effect_tasks`` is provided, only tasks in that set get
    ordering edges between them. Otherwise all tasks are chained.
    """
    if side_effect_tasks is not None:
        ordered = [t for t in tasks if t.task_id in side_effect_tasks]
    else:
        ordered = list(tasks)

    return tuple(
        Dependency(
            producer=ordered[index].task_id,
            consumer=ordered[index + 1].task_id,
            kind=DependencyKind.ORDERING,
        )
        for index in range(len(ordered) - 1)
    )


def build_mixed_dependencies(
    tasks: tuple[Task, ...],
    *,
    side_effect_tasks: frozenset[int] | None = None,
) -> tuple[Dependency, ...]:
    """Build dataflow dependencies plus ordering edges for side-effect tasks.

    Dataflow edges carry their natural kind (data/storage/war).
    Ordering edges are added between side-effect tasks that have no
    dataflow edge between them.
    """
    dataflow = build_dataflow_dependencies(tasks)
    dataflow_pairs = {(d.producer, d.consumer) for d in dataflow}

    ordering_candidates = build_ordering_dependencies(
        tasks, side_effect_tasks=side_effect_tasks,
    )
    extra_ordering = tuple(
        dep for dep in ordering_candidates
        if (dep.producer, dep.consumer) not in dataflow_pairs
    )
    return dataflow + extra_ordering


def build_dataflow_dependencies(tasks: tuple[Task, ...]) -> tuple[Dependency, ...]:
    """Build conservative RAW/WAW/WAR edges from task args and directions.

    Every task must carry ``arg_directions``. If directions are unavailable,
    callers should keep using ``sequential_v0`` instead of this policy.

    Edges are classified by kind:
    - ``"data"``: RAW (read-after-write) — reader depends on prior writer
    - ``"storage"``: WAW (write-after-write) — writer depends on prior writer
    - ``"war"``: WAR (write-after-read) — writer depends on prior reader
    """
    _require_complete_directions(tasks)

    edges: dict[tuple[int, int], str] = {}
    last_writer: dict[object, int] = {}
    readers_since_write: dict[object, set[int]] = {}

    for task in tasks:
        reads, writes = _read_write_args(task)

        for access_key in reads:
            writer = last_writer.get(access_key)
            if writer is not None and writer != task.task_id:
                _add_edge(edges, writer, task.task_id, DependencyKind.DATA)
            readers_since_write.setdefault(access_key, set()).add(task.task_id)

        for access_key in writes:
            writer = last_writer.get(access_key)
            if writer is not None and writer != task.task_id:
                _add_edge(edges, writer, task.task_id, DependencyKind.STORAGE)
            for reader in readers_since_write.get(access_key, set()):
                if reader != task.task_id:
                    _add_edge(edges, reader, task.task_id, DependencyKind.WAR)
            last_writer[access_key] = task.task_id
            readers_since_write[access_key] = set()

    return tuple(
        Dependency(producer=producer, consumer=consumer, kind=kind)
        for (producer, consumer), kind in sorted(edges.items())
    )


def _add_edge(
    edges: dict[tuple[int, int], str],
    producer: int,
    consumer: int,
    kind: str,
) -> None:
    """Add a dependency edge, preferring more specific kinds on conflict."""
    key = (producer, consumer)
    existing = edges.get(key)
    if existing is None or _kind_priority(kind) > _kind_priority(existing):
        edges[key] = kind


_KIND_PRIORITY = {
    DependencyKind.ORDERING: 0,
    DependencyKind.WAR: 1,
    DependencyKind.STORAGE: 2,
    DependencyKind.DATA: 3,
}


def _kind_priority(kind: str) -> int:
    return _KIND_PRIORITY.get(kind, 0)


def supports_dataflow_dependencies(tasks: tuple[Task, ...]) -> bool:
    """Return whether all tasks carry enough direction data for dataflow_v0."""
    return dataflow_dependency_fallback_code(tasks) is None


def dataflow_dependency_fallback_code(tasks: tuple[Task, ...]) -> FallbackCode | None:
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
    code = dataflow_dependency_fallback_code(tasks)
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
        normalized = normalize_direction(direction)
        if normalized in IGNORED_DIRECTIONS:
            continue
        access_key = storage_key if storage_key is not None else arg
        if normalized in READ_DIRECTIONS:
            reads.add(access_key)
        if normalized in WRITE_DIRECTIONS:
            writes.add(access_key)
    return reads, writes


def _storage_keys(task: Task) -> tuple[object | None, ...]:
    if task.arg_storage_keys:
        return task.arg_storage_keys
    return (None,) * len(task.args)


def derive_storage_effects(task: Task) -> tuple["StorageEffect", ...]:
    """Derive StorageEffect list from a Task's arg_directions and arg_storage_keys.

    v0.17 Phase 3 A3: Maps argument directions to storage access types:
    - input + has storage_key → StorageEffect(key, "read")
    - output/outputexisting + has storage_key → StorageEffect(key, "write")
    - inout + has storage_key → StorageEffect(key, "inplace_write")
    - scalar/nodep or no storage_key → no StorageEffect

    Returns empty tuple if Task has no arg_directions or arg_storage_keys.
    """
    from .score import StorageEffect

    if not task.arg_directions or not task.arg_storage_keys:
        return ()
    if len(task.arg_directions) != len(task.args):
        return ()
    if len(task.arg_storage_keys) != len(task.args):
        return ()

    effects: list[StorageEffect] = []
    for direction, storage_key in zip(task.arg_directions, task.arg_storage_keys):
        if storage_key is None:
            continue
        normalized = normalize_direction(direction)
        if normalized in IGNORED_DIRECTIONS:
            continue
        if normalized == "inout":
            effects.append(StorageEffect(buffer_id=str(storage_key), kind="inplace_write"))
        elif normalized in READ_DIRECTIONS:
            effects.append(StorageEffect(buffer_id=str(storage_key), kind="read"))
        elif normalized in WRITE_DIRECTIONS:
            effects.append(StorageEffect(buffer_id=str(storage_key), kind="write"))
    return tuple(effects)


def infer_side_effect(task: Task) -> bool:
    """Infer whether a Task is side-effecting from its arg_directions.

    v0.20 Phase 4 A2: Side-effect inference.

    A task is side-effecting if it has any output or inout arg_directions.
    Input-only and scalar/nodep tasks are not side-effecting.

    Returns False if Task has no arg_directions.
    """
    if not task.arg_directions:
        return False

    for direction in task.arg_directions:
        normalized = normalize_direction(direction)
        if normalized in WRITE_DIRECTIONS:
            return True
    return False


__all__ = [
    "DEPENDENCY_POLICY_DATAFLOW_V0",
    "DEPENDENCY_POLICY_SEQUENTIAL_V0",
    "build_dataflow_dependencies",
    "build_dependencies",
    "build_mixed_dependencies",
    "build_ordering_dependencies",
    "build_sequential_dependencies",
    "dataflow_dependency_fallback_code",
    "derive_storage_effects",
    "infer_side_effect",
    "supports_dataflow_dependencies",
]
