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


__all__ = [
    "ALIAS_ALIAS",
    "ALIAS_DISJOINT",
    "ALIAS_INPLACE",
    "ALIAS_VIEW",
    "AliasRelation",
    "analyze_aliases",
]
