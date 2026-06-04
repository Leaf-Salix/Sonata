# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Alias analysis for Sonata v0.3 storage model.

Determines alias relationships between storage keys: whether two keys
refer to the same underlying buffer (alias), a sub-view of it (view),
or share the buffer in an in-place operation (inplace).
"""

from dataclasses import dataclass
from typing import Any


ALIAS_DISJOINT = "disjoint"
ALIAS_ALIAS = "alias"
ALIAS_VIEW = "view"
ALIAS_INPLACE = "inplace"


@dataclass(frozen=True)
class AliasRelation:
    """Alias relationship between two storage keys."""

    key_a: str
    key_b: str
    relation: str

    @property
    def is_disjoint(self) -> bool:
        return self.relation == ALIAS_DISJOINT

    @property
    def shares_storage(self) -> bool:
        return self.relation in (ALIAS_ALIAS, ALIAS_VIEW, ALIAS_INPLACE)


def analyze_aliases(
    storage_keys: tuple[str, ...],
    alias_declarations: dict[str, str] | None = None,
    view_declarations: dict[str, str] | None = None,
    inplace_declarations: set[tuple[str, str]] | None = None,
) -> tuple[AliasRelation, ...]:
    """Analyze alias relationships between storage keys.

    ``alias_declarations`` maps a key to its canonical alias target.
    ``view_declarations`` maps a view key to its source key.
    ``inplace_declarations`` contains (key_a, key_b) pairs that share
    storage in-place.

    Keys with no declared relationship are assumed disjoint.
    """
    alias_map = alias_declarations or {}
    view_map = view_declarations or {}
    inplace_set = inplace_declarations or set()

    relations: list[AliasRelation] = []
    key_list = sorted(set(storage_keys))

    for i, key_a in enumerate(key_list):
        for key_b in key_list[i + 1:]:
            relation = _resolve_relation(key_a, key_b, alias_map, view_map, inplace_set)
            relations.append(AliasRelation(
                key_a=key_a, key_b=key_b, relation=relation,
            ))
    return tuple(relations)


def _resolve_relation(
    key_a: str,
    key_b: str,
    alias_map: dict[str, str],
    view_map: dict[str, str],
    inplace_set: set[tuple[str, str]],
) -> str:
    if key_a == key_b:
        return ALIAS_ALIAS

    if (key_a, key_b) in inplace_set or (key_b, key_a) in inplace_set:
        return ALIAS_INPLACE

    canon_a = alias_map.get(key_a, key_a)
    canon_b = alias_map.get(key_b, key_b)
    if canon_a == canon_b:
        return ALIAS_ALIAS

    source_a = view_map.get(key_a)
    source_b = view_map.get(key_b)
    if source_a is not None and source_a == key_b:
        return ALIAS_VIEW
    if source_b is not None and source_b == key_a:
        return ALIAS_VIEW
    if source_a is not None and source_b is not None and source_a == source_b:
        return ALIAS_VIEW

    return ALIAS_DISJOINT


def derive_aliases_from_tasks(
    tasks: tuple[Any, ...],
) -> tuple[AliasRelation, ...]:
    """Derive alias relationships from Tasks' arg_directions and arg_storage_keys.

    v0.20 Phase 2 A2: Task-based alias derivation.

    Rules:
    - Same buffer_id with inout → ALIAS_INPLACE
    - Same buffer_id with output + input (different tasks) → ALIAS_VIEW
    - Same buffer_id with multiple writes → ALIAS_ALIAS
    - Different buffer_ids → ALIAS_DISJOINT (not reported)

    Returns empty tuple if Tasks have no arg_directions or arg_storage_keys.
    """
    from .directions import READ_DIRECTIONS, WRITE_DIRECTIONS, normalize_direction

    # Collect per-task buffer access: buffer_id → list of (task_id, access_type)
    buffer_accesses: dict[str, list[tuple[int, str]]] = {}

    for task in tasks:
        if not task.arg_directions or not task.arg_storage_keys:
            continue
        for direction, storage_key in zip(task.arg_directions, task.arg_storage_keys):
            if storage_key is None:
                continue
            normalized = normalize_direction(direction)
            if normalized in ("scalar", "nodep"):
                continue
            key = str(storage_key)
            if key not in buffer_accesses:
                buffer_accesses[key] = []
            if normalized == "inout":
                buffer_accesses[key].append((task.task_id, "inplace"))
            elif normalized in READ_DIRECTIONS:
                buffer_accesses[key].append((task.task_id, "read"))
            elif normalized in WRITE_DIRECTIONS:
                buffer_accesses[key].append((task.task_id, "write"))

    # Build alias relations from buffer access patterns
    relations: list[AliasRelation] = []

    for buffer_id, accesses in buffer_accesses.items():
        task_ids = set(tid for tid, _ in accesses)
        access_types = set(atype for _, atype in accesses)

        if len(task_ids) <= 1:
            # Single task — no inter-task alias
            continue

        # Multiple tasks access same buffer
        if "inplace" in access_types:
            relations.append(AliasRelation(key_a=buffer_id, key_b=buffer_id, relation=ALIAS_INPLACE))
        elif "write" in access_types and "read" in access_types:
            relations.append(AliasRelation(key_a=buffer_id, key_b=buffer_id, relation=ALIAS_VIEW))
        elif "write" in access_types:
            relations.append(AliasRelation(key_a=buffer_id, key_b=buffer_id, relation=ALIAS_ALIAS))

    return tuple(relations)


__all__ = [
    "ALIAS_ALIAS",
    "ALIAS_DISJOINT",
    "ALIAS_INPLACE",
    "ALIAS_VIEW",
    "AliasRelation",
    "analyze_aliases",
    "derive_aliases_from_tasks",
]
