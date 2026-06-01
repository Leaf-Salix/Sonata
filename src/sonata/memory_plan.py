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

from dataclasses import dataclass, field
from typing import Any

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


__all__ = [
    "BufferAllocation",
    "MemoryPlan",
    "compute_conflict_matrix",
    "plan_memory",
]
