# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Small data model for Sonata static execution plans.

The model is intentionally pure Python and independent from PyPTO IR objects so
early analysis and tests can evolve without coupling to C++ bindings.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RuntimeTarget:
    """Runtime metadata emitted into generated ``kernel_config.py``."""

    runtime: str = "tensormap_and_ringbuffer"
    function_name: str = "aicpu_orchestration_entry"
    aicpu_thread_num: int | None = 4
    config_comment: tuple[str, ...] = (
        "# Runtime configuration for tensormap_and_ringbuffer.",
        "# This runtime requires 4 AICPU threads (3 schedulers + 1 orchestrator on thread 3).",
    )


DEFAULT_RUNTIME_TARGET = RuntimeTarget()


@dataclass(frozen=True)
class ShapeAssumption:
    """Static shape fact required for a score to remain valid."""

    symbol: str
    dims: tuple[int, ...]


@dataclass(frozen=True)
class Task:
    """One precomputed runtime task in a Sonata score."""

    task_id: int
    func_id: int
    core_type: str
    args: tuple[Any, ...] = ()
    arg_directions: tuple[str, ...] = ()
    arg_storage_keys: tuple[Any | None, ...] = ()
    name: str | None = None


@dataclass(frozen=True)
class Dependency:
    """Explicit edge between two precomputed tasks."""

    producer: int
    consumer: int


@dataclass(frozen=True)
class FallbackReason:
    """Structured explanation for why a score or region is ineligible.

    ``code`` is a best-effort slug derived from ``message`` and may change if
    the message wording changes. Do not use it as a stable routing or filtering
    key.
    """

    code: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class EligibilityResult:
    """Result of checking whether a score or IR region can use Sonata."""

    eligible: bool
    score: "Score | None" = None
    reasons: tuple[str, ...] = ()
    reason_details: tuple[FallbackReason, ...] = ()

    @classmethod
    def accept(cls, score: "Score") -> "EligibilityResult":
        """Build an eligible result for ``score``."""
        return cls(eligible=True, score=score)

    @classmethod
    def reject(cls, *reasons: str) -> "EligibilityResult":
        """Build an ineligible result with one or more explanatory reasons."""
        return cls(
            eligible=False,
            reasons=tuple(reasons),
            reason_details=tuple(FallbackReason(code=_reason_code(reason), message=reason) for reason in reasons),
        )


@dataclass(frozen=True)
class Score:
    """Inspectible static execution plan emitted before target-specific codegen."""

    name: str
    runtime_target: RuntimeTarget
    tasks: tuple[Task, ...] = ()
    dependencies: tuple[Dependency, ...] = ()
    shape_assumptions: tuple[ShapeAssumption, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def task_count(self) -> int:
        """Return the number of tasks in the score."""
        return len(self.tasks)

    def dependency_count(self) -> int:
        """Return the number of explicit dependency edges in the score."""
        return len(self.dependencies)

    def validate(self) -> EligibilityResult:
        """Validate basic score consistency for static planning."""
        reasons: list[str] = []
        task_ids = [task.task_id for task in self.tasks]
        task_id_set = set(task_ids)

        if not self.name:
            reasons.append("score name must not be empty")
        if len(task_ids) != len(task_id_set):
            reasons.append("task ids must be unique")

        for task in self.tasks:
            reasons.extend(_validate_task(task))

        for dep in self.dependencies:
            if dep.producer not in task_id_set:
                reasons.append(f"dependency producer is unknown: {dep.producer}")
            if dep.consumer not in task_id_set:
                reasons.append(f"dependency consumer is unknown: {dep.consumer}")
            if dep.producer == dep.consumer:
                reasons.append(f"dependency cannot be a self-edge: {dep.producer}")

        cycle = _find_cycle(task_id_set, self.dependencies)
        if cycle is not None:
            reasons.append(f"dependency graph must be acyclic, found cycle: {_format_cycle(cycle)}")

        shape_symbols: set[str] = set()
        for shape in self.shape_assumptions:
            symbol_seen = shape.symbol in shape_symbols
            if not symbol_seen:
                shape_symbols.add(shape.symbol)

            if not shape.symbol and not symbol_seen:
                reasons.append("shape assumption symbol must not be empty")
                continue
            if symbol_seen:
                reasons.append(f"shape assumption symbol must be unique: {_shape_symbol_label(shape.symbol)}")
                continue
            reasons.extend(_validate_shape_dims(shape.symbol, shape.dims))

        if reasons:
            return EligibilityResult.reject(*reasons)
        return EligibilityResult.accept(self)


def _find_cycle(task_ids: set[int], dependencies: tuple[Dependency, ...]) -> tuple[int, ...] | None:
    successors: dict[int, list[int]] = {task_id: [] for task_id in task_ids}
    for dep in dependencies:
        if dep.producer != dep.consumer and dep.producer in task_ids and dep.consumer in task_ids:
            successors[dep.producer].append(dep.consumer)

    visiting: set[int] = set()
    visited: set[int] = set()
    stack: list[int] = []

    for task_id in sorted(task_ids):
        cycle = _visit_for_cycle(task_id, successors, visiting, visited, stack)
        if cycle is not None:
            return cycle
    return None


def _validate_task(task: Task) -> list[str]:
    reasons: list[str] = []
    if task.task_id < 0:
        reasons.append(f"task id must be non-negative: {task.task_id}")
    if task.func_id < 0:
        reasons.append(f"task {task.task_id} func_id must be non-negative")
    if task.core_type not in {"aic", "aiv", "mixed"}:
        reasons.append(f"task {task.task_id} has unsupported core_type: {task.core_type}")
    if task.arg_directions and len(task.arg_directions) != len(task.args):
        reasons.append(
            f"task {task.task_id} arg_directions size {len(task.arg_directions)} "
            f"does not match args size {len(task.args)}"
        )
    if task.arg_storage_keys and len(task.arg_storage_keys) != len(task.args):
        reasons.append(
            f"task {task.task_id} arg_storage_keys size {len(task.arg_storage_keys)} "
            f"does not match args size {len(task.args)}"
        )
    return reasons


def is_static_shape_dim(dim: Any) -> bool:
    """Return whether ``dim`` is a positive concrete shape dimension."""
    return isinstance(dim, int) and not isinstance(dim, bool) and dim > 0


def _validate_shape_dims(symbol: str, dims: tuple[int, ...]) -> list[str]:
    reasons: list[str] = []
    seen: set[str] = set()
    for dim in dims:
        reason = _shape_dim_rejection_reason(symbol, dim)
        if reason is not None and reason not in seen:
            seen.add(reason)
            reasons.append(reason)
    return reasons


def _shape_dim_rejection_reason(symbol: str, dim: Any) -> str | None:
    if is_static_shape_dim(dim):
        return None
    if not isinstance(dim, int) or isinstance(dim, bool):
        return f"shape assumption {symbol} has non-integer dimension"
    if dim < 0:
        return f"shape assumption {symbol} has negative dimension"
    return f"shape assumption {symbol} has zero dimension"


def _shape_symbol_label(symbol: str) -> str:
    return symbol or "<empty>"


def _reason_code(reason: str) -> str:
    code = "".join(ch.lower() if ch.isalnum() else "_" for ch in reason).strip("_")
    return "_".join(part for part in code.split("_") if part) or "fallback"


def _visit_for_cycle(
    task_id: int,
    successors: dict[int, list[int]],
    visiting: set[int],
    visited: set[int],
    stack: list[int],
) -> tuple[int, ...] | None:
    if task_id in visited:
        return None
    if task_id in visiting:
        start = stack.index(task_id)
        return tuple(stack[start:] + [task_id])

    visiting.add(task_id)
    stack.append(task_id)
    for successor in sorted(successors[task_id]):
        cycle = _visit_for_cycle(successor, successors, visiting, visited, stack)
        if cycle is not None:
            return cycle
    stack.pop()
    visiting.remove(task_id)
    visited.add(task_id)
    return None


def _format_cycle(cycle: tuple[int, ...]) -> str:
    return " -> ".join(str(task_id) for task_id in cycle)
