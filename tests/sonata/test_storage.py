# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any

import pytest
from sonata.storage import (
    arg_storage_keys,
    call_write_storage_keys,
    collect_call_output_vars,
    collect_storage_keys,
    propagate_call_output_storage,
    storage_key,
)


@dataclass(frozen=True)
class Var:
    name_hint: str
    unique_id: int | None = None


@dataclass
class Call:
    op: str
    args: tuple[Any, ...] = ()
    arg_directions: tuple[str, ...] = ()


@dataclass
class AssignStmt:
    var: Any
    value: Any


@dataclass
class TupleGetItemExpr:
    tuple: Any
    index: int


@dataclass
class Function:
    params: tuple[Any, ...] = ()
    body: tuple[Any, ...] = ()


def test_collect_storage_keys_records_params_and_tensor_allocations() -> None:
    x = Var("x", unique_id=1)
    local = Var("local", unique_id=2)
    func = Function(params=(x,), body=(AssignStmt(local, Call("tensor.create")),))

    keys = _collect(func)

    assert keys[1] == "param:1:x"
    assert keys[2] == "alloc:2:local"


def test_collect_storage_keys_propagates_call_output_from_write_arg() -> None:
    x = Var("x", unique_id=1)
    local = Var("local", unique_id=2)
    result = Var("result", unique_id=3)
    call = Call("kernel", args=(x, local), arg_directions=("Input", "OutputExisting"))
    func = Function(
        params=(x,),
        body=(
            AssignStmt(local, Call("tensor.create")),
            AssignStmt(result, call),
        ),
    )

    keys = _collect(func)

    assert keys[2] == "alloc:2:local"
    assert keys[3] == "alloc:2:local"


def test_collect_storage_keys_propagates_tuple_get_elements_from_write_args() -> None:
    x = Var("x", unique_id=1)
    out0 = Var("out0", unique_id=2)
    out1 = Var("out1", unique_id=3)
    tuple_tmp = Var("_tuple_tmp", unique_id=4)
    first = Var("first", unique_id=5)
    second = Var("second", unique_id=6)
    call = Call(
        "kernel",
        args=(x, out0, out1),
        arg_directions=("Input", "OutputExisting", "OutputExisting"),
    )
    func = Function(
        params=(x,),
        body=(
            AssignStmt(out0, Call("tensor.create")),
            AssignStmt(out1, Call("tensor.create")),
            AssignStmt(tuple_tmp, call),
            AssignStmt(first, TupleGetItemExpr(tuple_tmp, 0)),
            AssignStmt(second, TupleGetItemExpr(tuple_tmp, 1)),
        ),
    )

    keys = _collect(func)

    assert keys[5] == "alloc:2:out0"
    assert keys[6] == "alloc:3:out1"


def test_collect_storage_keys_skips_builtin_non_alloc_calls() -> None:
    local = Var("local", unique_id=1)
    result = Var("result", unique_id=2)
    func = Function(
        body=(
            AssignStmt(local, Call("tensor.create")),
            AssignStmt(result, Call("tensor.slice", args=(local,), arg_directions=("Input",))),
        )
    )

    keys = _collect(func)

    assert keys == {1: "alloc:1:local"}


def test_collect_call_output_vars_tracks_assigned_calls() -> None:
    lhs = Var("lhs", unique_id=1)
    call = Call("kernel")
    func = Function(body=(AssignStmt(lhs, call),))

    outputs = collect_call_output_vars(func, walk=_walk, kind=_kind)

    assert outputs == {id(call): lhs}


def test_arg_storage_keys_projects_known_and_unknown_args() -> None:
    known = Var("known", unique_id=1)
    unknown = Var("unknown", unique_id=2)
    call = Call("kernel", args=(known, unknown))

    keys = arg_storage_keys(call, {1: "buffer:known"})

    assert keys == ("buffer:known", None)
    assert storage_key(unknown, {1: "buffer:known"}) is None


def test_propagate_call_output_storage_accepts_normalized_direction_names() -> None:
    out = Var("out", unique_id=1)
    result = Var("result", unique_id=2)
    call = Call("kernel", args=(out,), arg_directions=("output_existing",))
    keys = {1: "buffer:out"}

    propagate_call_output_storage(result, call, keys, arg_directions=_arg_directions)

    assert keys[2] == "buffer:out"


def test_call_write_storage_keys_returns_write_args_only() -> None:
    x = Var("x", unique_id=1)
    out0 = Var("out0", unique_id=2)
    out1 = Var("out1", unique_id=3)
    call = Call("kernel", args=(x, out0, out1), arg_directions=("Input", "OutputExisting", "InOut"))

    keys = call_write_storage_keys(
        call,
        {1: "param:1:x", 2: "alloc:2:out0", 3: "alloc:3:out1"},
        arg_directions=_arg_directions,
    )

    assert keys == ("alloc:2:out0", "alloc:3:out1")


def test_propagate_call_output_storage_ignores_read_only_args() -> None:
    x = Var("x", unique_id=1)
    result = Var("result", unique_id=2)
    call = Call("kernel", args=(x,), arg_directions=("Input",))
    keys = {1: "param:1:x"}

    propagate_call_output_storage(result, call, keys, arg_directions=_arg_directions)

    assert 2 not in keys


def _collect(node: Any) -> dict[int, str]:
    return collect_storage_keys(
        node,
        walk=_walk,
        kind=_kind,
        call_name=_call_name,
        is_builtin_call=_is_builtin_call,
        arg_name=_arg_name,
        arg_directions=_arg_directions,
    )


def _walk(node: Any):
    yield node
    for field in ("params", "body", "args", "value"):
        value = getattr(node, field, None)
        if isinstance(value, tuple):
            for item in value:
                yield from _walk(item)
        elif value is not None and not isinstance(value, (str, int)):
            yield from _walk(value)


def _kind(node: Any) -> str:
    return type(node).__name__


def _call_name(node: Any) -> str | None:
    return node.op if isinstance(node, Call) else None


def _is_builtin_call(name: str) -> bool:
    return name.startswith(("tile.", "tensor.", "system.", "array."))


def _arg_name(node: Any) -> str:
    return node.name_hint if isinstance(node, Var) else _kind(node)


def _arg_directions(node: Any) -> tuple[str, ...]:
    return node.arg_directions if isinstance(node, Call) else ()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
