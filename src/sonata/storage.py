# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Structural storage-key extraction for Sonata tasks.

This module uses Python-visible fields only. It intentionally avoids importing
PyPTO IR classes so the experimental Sonata score layer stays easy to rebase.
"""

from collections.abc import Callable, Iterable
from typing import Any

_WRITE_DIRECTIONS = {"output", "outputexisting", "inout"}

STORAGE_COVERAGE_WARN_THRESHOLD = 0.5
STORAGE_COVERAGE_REJECT_THRESHOLD = 0.0


def collect_storage_keys(
    node: Any,
    *,
    walk: Callable[[Any], Iterable[Any]],
    kind: Callable[[Any], str],
    call_name: Callable[[Any], str | None],
    is_builtin_call: Callable[[str], bool],
    arg_name: Callable[[Any], str],
    arg_directions: Callable[[Any], tuple[str, ...]],
) -> dict[int, str]:
    """Return structural storage keys keyed by stable Var identity."""
    storage_keys: dict[int, str] = {}
    tuple_element_keys: dict[int, tuple[str | None, ...]] = {}
    for param in getattr(node, "params", ()):
        key = _var_storage_key(param, "param", kind=kind, arg_name=arg_name)
        if key is not None:
            storage_keys[_var_identity(param)] = key

    for child in walk(node):
        if kind(child) != "AssignStmt":
            continue
        var = getattr(child, "var", None)
        value = getattr(child, "value", None)
        if var is None or kind(value) != "Call":
            if var is not None and kind(value) == "TupleGetItemExpr":
                _propagate_tuple_get_storage(var, value, storage_keys, tuple_element_keys)
            continue
        _collect_call_assignment_storage(
            var,
            value,
            storage_keys,
            tuple_element_keys,
            kind=kind,
            call_name=call_name,
            is_builtin_call=is_builtin_call,
            arg_name=arg_name,
            arg_directions=arg_directions,
        )
    return storage_keys


def collect_call_output_vars(
    node: Any,
    *,
    walk: Callable[[Any], Iterable[Any]],
    kind: Callable[[Any], str],
) -> dict[int, Any]:
    """Return assignment LHS variables keyed by assigned Call object identity."""
    outputs: dict[int, Any] = {}
    for child in walk(node):
        if kind(child) != "AssignStmt":
            continue
        value = getattr(child, "value", None)
        if kind(value) == "Call":
            outputs[id(value)] = getattr(child, "var", None)
    return outputs


def arg_storage_keys(call: Any, storage_keys: dict[int, str]) -> tuple[str | None, ...]:
    """Return storage keys for each positional call argument when known."""
    return tuple(storage_key(arg, storage_keys) for arg in getattr(call, "args", ()))


def propagate_call_output_storage(
    output_var: Any,
    call: Any,
    storage_keys: dict[int, str],
    *,
    arg_directions: Callable[[Any], tuple[str, ...]],
) -> None:
    """Propagate the first known write-arg storage key to an assigned call output."""
    for arg, direction in zip(getattr(call, "args", ()), arg_directions(call), strict=False):
        if _normalize_direction(direction) not in _WRITE_DIRECTIONS:
            continue
        key = storage_key(arg, storage_keys)
        if key is not None:
            storage_keys[_var_identity(output_var)] = key
            return


def call_write_storage_keys(
    call: Any,
    storage_keys: dict[int, str],
    *,
    arg_directions: Callable[[Any], tuple[str, ...]],
) -> tuple[str | None, ...]:
    """Return storage keys for write-like call args in positional order."""
    keys: list[str | None] = []
    for arg, direction in zip(getattr(call, "args", ()), arg_directions(call), strict=False):
        if _normalize_direction(direction) in _WRITE_DIRECTIONS:
            keys.append(storage_key(arg, storage_keys))
    return tuple(keys)


def storage_key(arg: Any, storage_keys: dict[int, str]) -> str | None:
    """Return the known storage key for ``arg`` if one has been collected."""
    return storage_keys.get(_var_identity(arg))


def _var_storage_key(
    var: Any,
    prefix: str,
    *,
    kind: Callable[[Any], str],
    arg_name: Callable[[Any], str],
) -> str | None:
    name = arg_name(var)
    if name == kind(var):
        return None
    unique_id = getattr(var, "unique_id", None)
    if isinstance(unique_id, int):
        return f"{prefix}:{unique_id}:{name}"
    return f"{prefix}:{name}"


def _collect_call_assignment_storage(
    output_var: Any,
    call: Any,
    storage_keys: dict[int, str],
    tuple_element_keys: dict[int, tuple[str | None, ...]],
    *,
    kind: Callable[[Any], str],
    call_name: Callable[[Any], str | None],
    is_builtin_call: Callable[[str], bool],
    arg_name: Callable[[Any], str],
    arg_directions: Callable[[Any], tuple[str, ...]],
) -> None:
    name = call_name(call)
    if name == "tensor.create":
        key = _var_storage_key(output_var, "alloc", kind=kind, arg_name=arg_name)
        if key is not None:
            storage_keys[_var_identity(output_var)] = key
    elif name is not None and not is_builtin_call(name):
        keys = call_write_storage_keys(call, storage_keys, arg_directions=arg_directions)
        tuple_element_keys[_var_identity(output_var)] = keys
        propagate_call_output_storage(output_var, call, storage_keys, arg_directions=arg_directions)


def _propagate_tuple_get_storage(
    output_var: Any,
    tuple_get: Any,
    storage_keys: dict[int, str],
    tuple_element_keys: dict[int, tuple[str | None, ...]],
) -> None:
    tuple_expr = getattr(tuple_get, "tuple", None)
    index = getattr(tuple_get, "index", None)
    if not isinstance(index, int):
        return
    keys = tuple_element_keys.get(_var_identity(tuple_expr))
    if keys is None or index < 0 or index >= len(keys):
        return
    key = keys[index]
    if key is not None:
        storage_keys[_var_identity(output_var)] = key


def _var_identity(var: Any) -> int:
    unique_id = getattr(var, "unique_id", None)
    if isinstance(unique_id, int):
        return unique_id
    return id(var)


def _normalize_direction(direction: str) -> str:
    return "".join(ch for ch in str(direction).lower() if ch.isalnum())


__all__ = [
    "arg_storage_keys",
    "call_write_storage_keys",
    "collect_call_output_vars",
    "collect_storage_keys",
    "propagate_call_output_storage",
    "storage_key",
]
