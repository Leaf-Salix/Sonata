# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Audit metadata helpers for Sonata scores.

The helpers operate on Sonata's pure-Python task model and structural entry
roots. They intentionally stay out of dependency construction so audit data
can explain a score without changing its scheduling semantics.
"""

from typing import Any

from .fallback import FallbackCode
from .score import Task

_MEMORY_DIRECTIONS = {"input", "inout", "output", "outputexisting"}


def build_score_metadata(
    extraction_roots: tuple[Any, ...],
    tasks: tuple[Task, ...],
    entry_name: str | None,
    dependency_policy: str,
    requested_dependency_policy: str,
    *,
    fallback_code: FallbackCode | None = None,
) -> dict[str, Any]:
    """Build explanatory metadata for an extracted Sonata score."""
    metadata: dict[str, Any] = {
        "extractor": "structural_v0",
        "dependency_policy": dependency_policy,
    }
    metadata.update(build_task_storage_metadata(tasks))
    if fallback_code is not None:
        metadata["requested_dependency_policy"] = requested_dependency_policy
        metadata["dependency_policy_fallback_reason"] = fallback_code.value
    entry_names = tuple(
        name for root in extraction_roots if isinstance((name := getattr(root, "name", None)), str)
    )
    if entry_name is not None:
        metadata["entry_name"] = entry_name
    elif len(entry_names) > 1:
        metadata["entry_policy"] = "all_orchestration"
        metadata["entry_names"] = entry_names
    elif len(entry_names) == 1:
        metadata["entry_name"] = entry_names[0]
    return metadata


def build_task_storage_metadata(tasks: tuple[Task, ...]) -> dict[str, Any]:
    """Build storage-key coverage and ``NoDep`` audit metadata for tasks."""
    total_args = 0
    known_storage_args = 0
    memory_args = 0
    known_memory_args = 0
    nodep_args: list[dict[str, Any]] = []
    unknown_memory_args: list[dict[str, Any]] = []

    for task in tasks:
        storage_keys = _task_storage_keys(task)
        directions = _task_directions(task)
        for index, arg in enumerate(task.args):
            direction = directions[index] if index < len(directions) else "unknown"
            storage_key = storage_keys[index] if index < len(storage_keys) else None
            total_args += 1
            if storage_key is not None:
                known_storage_args += 1
            normalized = _normalize_direction(direction)
            arg_record = _task_arg_record(task, index, arg, storage_key)
            if normalized == "nodep":
                nodep_args.append(arg_record)
            if normalized in _MEMORY_DIRECTIONS:
                memory_args += 1
                if storage_key is not None:
                    known_memory_args += 1
                else:
                    unknown_memory_args.append(arg_record)

    metadata: dict[str, Any] = {
        "storage_key_coverage": {
            "known": known_storage_args,
            "unknown": total_args - known_storage_args,
            "total": total_args,
        },
        "memory_storage_key_coverage": {
            "known": known_memory_args,
            "unknown": memory_args - known_memory_args,
            "total": memory_args,
        },
    }
    if nodep_args:
        metadata["nodep_args"] = tuple(nodep_args)
    if unknown_memory_args:
        metadata["unknown_memory_storage_args"] = tuple(unknown_memory_args)
    return metadata


def _task_storage_keys(task: Task) -> tuple[Any | None, ...]:
    if task.arg_storage_keys:
        return task.arg_storage_keys
    return (None,) * len(task.args)


def _task_directions(task: Task) -> tuple[str, ...]:
    if task.arg_directions:
        return task.arg_directions
    return ("unknown",) * len(task.args)


def _task_arg_record(task: Task, index: int, arg: Any, storage_key: Any | None) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "task_name": task.name,
        "arg_index": index,
        "arg": arg,
        "storage_key": storage_key,
    }


def _normalize_direction(direction: str) -> str:
    return "".join(ch for ch in str(direction).lower() if ch.isalnum())


__all__ = [
    "build_score_metadata",
    "build_task_storage_metadata",
]
