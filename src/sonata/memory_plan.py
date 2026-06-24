# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Graph-level memory planning for Sonata v0.3.

Implements a greedy-first-fit (by size) 1D strip-packing allocator.
Buffers are sorted by size descending and placed at the first available
offset that does not conflict with overlapping lifetimes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
import warnings

from .liveness import BufferLifetime, find_conflicts


@dataclass(frozen=True)
class BufferAllocation:
    """One buffer's memory allocation."""

    storage_key: str
    offset: int
    size: int

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass(frozen=True)
class MemoryPlan:
    """Memory layout plan for a Score's buffers."""

    allocations: tuple[BufferAllocation, ...] = ()
    peak_memory: int = 0
    # v0.11 Phase 2 C1-C2: Schema extension
    solver_type: str = "greedy"
    conflict_matrix_hash: str | None = None

    def total_allocated(self) -> int:
        return sum(a.size for a in self.allocations)

    def by_key(self, storage_key: str) -> BufferAllocation | None:
        for alloc in self.allocations:
            if alloc.storage_key == storage_key:
                return alloc
        return None


def plan_memory(
    lifetimes: tuple[BufferLifetime, ...],
    buffer_sizes: dict[str, int],
) -> MemoryPlan:
    """Plan memory layout using greedy-first-fit by size.

    ``buffer_sizes`` maps storage keys to their byte sizes.
    Buffers with overlapping lifetimes cannot share memory.
    """
    if not lifetimes:
        return MemoryPlan()

    conflicts = find_conflicts(lifetimes)
    conflict_pairs: set[tuple[str, str]] = set()
    for c in conflicts:
        conflict_pairs.add((c.key_a, c.key_b))
        conflict_pairs.add((c.key_b, c.key_a))

    sorted_lifetimes = sorted(
        lifetimes,
        key=lambda lt: buffer_sizes.get(lt.storage_key, 0),
        reverse=True,
    )

    allocations: list[BufferAllocation] = []
    for lt in sorted_lifetimes:
        size = buffer_sizes.get(lt.storage_key, 0)
        offset = _find_first_fit(
            size, lt, allocations, lifetimes, conflict_pairs,
        )
        allocations.append(BufferAllocation(
            storage_key=lt.storage_key,
            offset=offset,
            size=size,
        ))

    peak = max((a.end for a in allocations), default=0)
    return MemoryPlan(
        allocations=tuple(sorted(allocations, key=lambda a: a.storage_key)),
        peak_memory=peak,
    )


def _find_first_fit(
    size: int,
    lifetime: BufferLifetime,
    existing: list[BufferAllocation],
    all_lifetimes: tuple[BufferLifetime, ...],
    conflict_pairs: set[tuple[str, str]],
) -> int:
    """Find the first offset where this buffer can fit without conflict."""
    if size == 0:
        return 0

    lifetime_map = {lt.storage_key: lt for lt in all_lifetimes}

    conflicting = []
    for alloc in existing:
        other_lt = lifetime_map.get(alloc.storage_key)
        if other_lt is not None and lifetime.overlaps(other_lt):
            conflicting.append(alloc)

    if not conflicting:
        return 0

    conflicting.sort(key=lambda a: a.offset)
    candidate = 0
    for alloc in conflicting:
        if candidate + size <= alloc.offset:
            break
        if candidate < alloc.end:
            candidate = alloc.end

    return candidate


def compute_conflict_matrix(
    lifetimes: list[BufferLifetime],
    *,
    stream_ids: dict[str, int] | None = None,
) -> list[list[bool]]:
    """Build an NxN boolean conflict matrix from buffer lifetimes.

    v0.11 Phase 2 A1-A3: Conflict matrix computation with stream-aware
    and in-place handling.

    ``matrix[i][j]`` is True when lifetimes[i] and lifetimes[j] overlap
    AND are on the same stream (or no stream info is given).

    Args:
        lifetimes: Buffer lifetime intervals.
        stream_ids: Optional mapping of storage_key -> stream_id.
            Buffers on different streams never conflict.

    Returns:
        NxN boolean matrix (list of lists).
    """
    n = len(lifetimes)
    matrix = [[False] * n for _ in range(n)]

    for i in range(n):
        matrix[i][i] = False  # no self-conflict by default
        for j in range(i + 1, n):
            # Stream-aware: skip if different streams
            if stream_ids is not None:
                si = stream_ids.get(lifetimes[i].storage_key, 0)
                sj = stream_ids.get(lifetimes[j].storage_key, 0)
                if si != sj:
                    continue

            if lifetimes[i].overlaps(lifetimes[j]):
                matrix[i][j] = True
                matrix[j][i] = True

    return matrix


# ---------------------------------------------------------------------------
# v0.11 Phase 2 B: Constraint Solver
# ---------------------------------------------------------------------------


class ConstraintSolver(ABC):
    """ABC for memory constraint solvers.

    v0.11 Phase 2 B1: Abstract interface for memory planning solvers.
    Implementations must assign non-overlapping offsets to conflicting
    buffers while respecting size and device memory constraints.
    """

    @abstractmethod
    def solve(
        self,
        conflict_matrix: list[list[bool]],
        sizes: list[int],
        device_memory_limit: int | None = None,
    ) -> MemoryPlan:
        """Assign offsets to buffers subject to conflict constraints.

        Args:
            conflict_matrix: NxN boolean matrix (True = cannot share memory).
            sizes: Buffer sizes in bytes (length N).
            device_memory_limit: Optional upper bound on total memory.

        Returns:
            MemoryPlan with offset assignments.
        """


class GreedySolver(ConstraintSolver):
    """Greedy first-fit-decreasing solver.

    v0.11 Phase 2 B2: Fallback strategy when the primary solver times out.
    Sorts buffers by size descending and places each at the first offset
    that doesn't conflict with already-placed buffers.
    """

    def solve(
        self,
        conflict_matrix: list[list[bool]],
        sizes: list[int],
        device_memory_limit: int | None = None,
    ) -> MemoryPlan:
        n = len(sizes)
        if n == 0:
            return MemoryPlan(allocations=(), peak_memory=0, solver_type="greedy")

        order = sorted(range(n), key=lambda i: sizes[i], reverse=True)
        offsets: dict[int, int] = {}

        for idx in order:
            size = sizes[idx]
            offset = 0
            placed = False
            while not placed:
                conflict = False
                for other_idx, other_offset in offsets.items():
                    if conflict_matrix[idx][other_idx]:
                        other_size = sizes[other_idx]
                        if offset < other_offset + other_size and other_offset < offset + size:
                            offset = other_offset + other_size
                            conflict = True
                            break
                if not conflict:
                    offsets[idx] = offset
                    placed = True

        allocations = tuple(
            BufferAllocation(storage_key=f"buf_{i}", offset=offsets[i], size=sizes[i])
            for i in range(n)
        )
        peak = max((a.offset + a.size for a in allocations), default=0)

        if device_memory_limit is not None and peak > device_memory_limit:
            raise MemoryLimitExceededError(
                f"Memory plan peak ({peak} bytes) exceeds device limit ({device_memory_limit} bytes)",
                peak_memory=peak,
                device_limit=device_memory_limit,
            )

        return MemoryPlan(allocations=allocations, peak_memory=peak, solver_type="greedy")


class DynamicShapeError(Exception):
    """Raised when dynamic shapes prevent memory planning."""


class MemoryLimitExceededError(Exception):
    """Raised when a memory plan exceeds the device memory limit."""

    def __init__(self, message: str, *, peak_memory: int, device_limit: int):
        super().__init__(message)
        self.peak_memory = peak_memory
        self.device_limit = device_limit


class MemoryPlanExceededError(Exception):
    """Raised when a memory plan exceeds the device memory limit."""

    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        super().__init__(
            f"Dynamic shape symbols {symbols} are not supported by the memory planner. "
            "Consider using static shapes or enabling runtime shape inference."
        )


def solve_memory(
    solver: ConstraintSolver,
    conflict_matrix: list[list[bool]],
    sizes: list[int],
    device_memory_limit: int | None = None,
    *,
    timeout_seconds: float = 1.0,
    fallback: ConstraintSolver | None = None,
) -> MemoryPlan:
    """Run a solver with timeout and optional greedy fallback.

    v0.11 Phase 2 B2: If the primary solver exceeds ``timeout_seconds``,
    falls back to ``fallback`` (default: GreedySolver).

    Args:
        solver: Primary solver to try first.
        conflict_matrix: NxN boolean conflict matrix.
        sizes: Buffer sizes.
        device_memory_limit: Optional memory cap.
        timeout_seconds: Max wall-clock time for primary solver.
        fallback: Solver to use on timeout (default: GreedySolver).

    Returns:
        MemoryPlan from whichever solver succeeded.
    """
    if fallback is None:
        fallback = GreedySolver()

    start = time.monotonic()
    try:
        result = solver.solve(conflict_matrix, sizes, device_memory_limit)
        elapsed = time.monotonic() - start
        if elapsed > timeout_seconds:
            warnings.warn(
                f"Primary solver took {elapsed:.2f}s (>{timeout_seconds}s), "
                "falling back to GreedySolver.",
                UserWarning,
                stacklevel=2,
            )
            return fallback.solve(conflict_matrix, sizes, device_memory_limit)
        return result
    except Exception:
        warnings.warn(
            "Primary solver failed, falling back to GreedySolver.",
            UserWarning,
            stacklevel=2,
        )
        return fallback.solve(conflict_matrix, sizes, device_memory_limit)


# ---------------------------------------------------------------------------
# v0.20 Phase 3: Buffer sharing legality check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharingLegalityResult:
    """Result of checking whether two buffers can legally share memory."""

    buffer_a: str
    buffer_b: str
    legal: bool
    reason: str  # "compatible", "different_space", "incompatible_dtype", etc.


def check_sharing_legality(
    lifetimes: tuple[BufferLifetime, ...],
    buffer_metadata: dict[str, dict[str, str]],
) -> tuple[SharingLegalityResult, ...]:
    """Check whether non-conflicting buffers can legally share memory.

    v0.20 Phase 3 A1: Legality check (aligned with PyPTO LegalizePTOBufferReuse).

    ``buffer_metadata`` maps storage_key → {"memory_space": str, "dtype": str}.
    Buffers must be in the same memory_space and have compatible dtypes to share.

    Only checks non-conflicting pairs (those that could potentially share).
    """
    from .liveness import find_conflicts

    conflicts = find_conflicts(lifetimes)
    conflict_set: set[tuple[str, str]] = set()
    for c in conflicts:
        conflict_set.add((c.key_a, c.key_b))
        conflict_set.add((c.key_b, c.key_a))

    results: list[SharingLegalityResult] = []
    keys = sorted({lt.storage_key for lt in lifetimes})

    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1:]:
            # Skip conflicting pairs — they can't share anyway
            if (key_a, key_b) in conflict_set:
                continue

            meta_a = buffer_metadata.get(key_a, {})
            meta_b = buffer_metadata.get(key_b, {})

            space_a = meta_a.get("memory_space", "unknown")
            space_b = meta_b.get("memory_space", "unknown")
            dtype_a = meta_a.get("dtype", "unknown")
            dtype_b = meta_b.get("dtype", "unknown")

            if space_a != space_b:
                results.append(SharingLegalityResult(
                    buffer_a=key_a, buffer_b=key_b,
                    legal=False, reason="different_space",
                ))
            elif dtype_a != dtype_b and dtype_a != "unknown" and dtype_b != "unknown":
                results.append(SharingLegalityResult(
                    buffer_a=key_a, buffer_b=key_b,
                    legal=False, reason="incompatible_dtype",
                ))
            else:
                results.append(SharingLegalityResult(
                    buffer_a=key_a, buffer_b=key_b,
                    legal=True, reason="compatible",
                ))

    return tuple(results)


__all__ = [
    "BufferAllocation",
    "ConstraintSolver",
    "DynamicShapeError",
    "GreedySolver",
    "MemoryLimitExceededError",
    "MemoryPlan",
    "SharingLegalityResult",
    "check_sharing_legality",
    "compute_conflict_matrix",
    "plan_memory",
    "solve_memory",
]
