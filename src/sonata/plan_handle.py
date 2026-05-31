# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PlanHandle and supporting types for Sonata v0.2 runtime contract.

PlanHandle carries the runtime artifact key information that was previously
embedded in Score. Score remains the computation identity; PlanHandle bridges
Score to a specific runtime target, adapter stage, and function registry.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .score import RuntimeTarget, Score


class GuardStatus(Enum):
    """Current guard evaluation status for a PlanHandle.
    
    Attributes:
        ALL_SATISFIED: All guard conditions are currently satisfied.
            The plan handle is valid and can be used.
        PARTIAL_FAILED: Some guards failed but not critical ones.
            May still be usable with soft invalidation strategy.
        ALL_FAILED: Critical guards failed. Plan handle must be invalidated
            and replan triggered from Score.
    """
    
    ALL_SATISFIED = "all_satisfied"
    PARTIAL_FAILED = "partial_failed"
    ALL_FAILED = "all_failed"


PLAN_HANDLE_SCHEMA_VERSION = 1
RUNTIME_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class FuncRegistryEntry:
    """One entry in the function registry."""

    name: str
    sonata_func_id: int
    runtime_func_id: int | None = None


@dataclass(frozen=True)
class FuncRegistry:
    """Explicit function name to runtime id mapping.

    Uses function name as the logical key rather than relying on
    extraction order, so that Sonata func_ids can be aligned with
    PyPTO codegen func_name_to_id independently.
    """

    entries: tuple[FuncRegistryEntry, ...] = ()

    def by_name(self, name: str) -> FuncRegistryEntry | None:
        for entry in self.entries:
            if entry.name == name:
                return entry
        return None

    def names(self) -> frozenset[str]:
        return frozenset(entry.name for entry in self.entries)

    def sonata_func_ids(self) -> dict[str, int]:
        return {entry.name: entry.sonata_func_id for entry in self.entries}

    def runtime_func_ids(self) -> dict[str, int | None]:
        return {entry.name: entry.runtime_func_id for entry in self.entries}

    def has_unbound_entries(self) -> bool:
        return any(entry.runtime_func_id is None for entry in self.entries)

    def bind_runtime_ids(
        self, func_name_to_id: dict[str, int]
    ) -> "FuncRegistry":
        """Return a new registry with runtime ids bound from a name-to-id map."""
        bound = []
        for entry in self.entries:
            runtime_id = func_name_to_id.get(entry.name, entry.runtime_func_id)
            bound.append(
                FuncRegistryEntry(
                    name=entry.name,
                    sonata_func_id=entry.sonata_func_id,
                    runtime_func_id=runtime_id,
                )
            )
        return FuncRegistry(entries=tuple(bound))

    @classmethod
    def from_score(cls, score: Score) -> "FuncRegistry":
        """Build a registry from Score tasks, using callee names as keys."""
        seen: dict[str, int] = {}
        for task in score.tasks:
            name = task.name
            if name is not None and name not in seen:
                seen[name] = task.func_id
        entries = tuple(
            FuncRegistryEntry(name=name, sonata_func_id=fid)
            for name, fid in sorted(seen.items())
        )
        return cls(entries=entries)


@dataclass(frozen=True)
class RuntimeArgBinding:
    """Maps a task arg from Sonata storage identity to a runtime handle.

    ``runtime_handle`` is opaque at the PlanHandle level. For positional
    binding it carries an integer index; for codegen-aware binding it may
    carry a runtime tensor/scalar handle.
    """

    task_id: int
    arg_index: int
    storage_key: str | None
    direction: str
    runtime_handle: Any


@dataclass(frozen=True)
class PlanHandle:
    """Runtime artifact key for a Sonata static plan.

    Bridges a Score (computation identity) to a specific runtime target,
    adapter stage, and function registry. PlanHandle is the primary carrier
    for runtime-specific metadata in v0.2+.
    
    Attributes:
        score_fingerprint: Canonical fingerprint of the computation (Score).
        runtime_target: Target runtime configuration (e.g., host_build_graph).
        source_adapter: Adapter that produced this PlanHandle.
        runtime_contract_version: Version of the runtime contract.
        func_registry: Function name to runtime ID mapping.
        arg_bindings: Runtime argument bindings for tasks.
        schema_version: PlanHandle schema version.
        metadata: Additional metadata dictionary.
        guard_status: Current evaluation status of guard conditions.
        critical_guards: Subset of guards deemed critical for fingerprinting.
    """

    score_fingerprint: str
    runtime_target: RuntimeTarget
    source_adapter: str
    runtime_contract_version: int = RUNTIME_CONTRACT_VERSION
    func_registry: FuncRegistry = field(default_factory=FuncRegistry)
    arg_bindings: tuple[RuntimeArgBinding, ...] = ()
    schema_version: int = PLAN_HANDLE_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    # Phase 4: Guard condition integration
    guard_status: GuardStatus = GuardStatus.ALL_SATISFIED
    critical_guards: tuple[Any, ...] = field(default_factory=tuple)  # GuardCondition instances

    @classmethod
    def from_score(
        cls,
        score: Score,
        *,
        source_adapter: str = "post_simplify",
        runtime_target: RuntimeTarget | None = None,
    ) -> "PlanHandle":
        """Build a minimal PlanHandle from a Score.

        Uses ``score_fingerprint()`` for the computation identity and
        ``FuncRegistry.from_score()`` for the function registry.
        """
        from .serialization import score_fingerprint

        target = runtime_target or score.runtime_target
        return cls(
            score_fingerprint=score_fingerprint(score),
            runtime_target=target,
            source_adapter=source_adapter,
            func_registry=FuncRegistry.from_score(score),
        )


__all__ = [
    "FuncRegistry",
    "FuncRegistryEntry",
    "PLAN_HANDLE_SCHEMA_VERSION",
    "PlanHandle",
    "RUNTIME_CONTRACT_VERSION",
    "RuntimeArgBinding",
    "GuardStatus",
]
