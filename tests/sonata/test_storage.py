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
from sonata.score import Task
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

    assert keys[1] == "param:x"
    assert keys[2] == "alloc:local"


def test_collect_storage_keys_disambiguates_duplicate_stable_names() -> None:
    first = Var("x", unique_id=1)
    second = Var("x", unique_id=2)
    third = Var("x", unique_id=3)
    func = Function(params=(first, second, third))

    keys = _collect(func)

    assert keys[1] == "param:x"
    assert keys[2] == "param:x@2"
    assert keys[3] == "param:x@3"


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

    assert keys[2] == "alloc:local"
    assert keys[3] == "alloc:local"


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

    assert keys[5] == "alloc:out0"
    assert keys[6] == "alloc:out1"


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

    assert keys == {1: "alloc:local"}


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


def test_storage_key_uses_unique_id_for_lookup_but_not_key_text() -> None:
    alias = Var("x_alias", unique_id=1)
    call = Call("kernel", args=(alias,))

    keys = arg_storage_keys(call, {1: "param:x"})

    assert storage_key(alias, {1: "param:x"}) == "param:x"
    assert keys == ("param:x",)


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
        {1: "param:x", 2: "alloc:out0", 3: "alloc:out1"},
        arg_directions=_arg_directions,
    )

    assert keys == ("alloc:out0", "alloc:out1")


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


class TestStorageCoverageThresholds:
    """Tests for the storage coverage eligibility check."""

    def _score_with_coverage(self, tasks, *, name="test"):
        from sonata.score import Score, RuntimeTarget
        from sonata.audit import build_score_metadata
        metadata = build_score_metadata((), tasks, None, "sequential_v0", "sequential_v0")
        return Score(name=name, runtime_target=RuntimeTarget(), tasks=tasks, metadata=metadata)

    def test_full_coverage_eligible_no_warnings(self) -> None:
        from sonata.eligibility import _check_storage_coverage

        tasks = (
            Task(task_id=0, func_id=0, core_type="aiv",
                 args=("a",), arg_directions=("input",), arg_storage_keys=("param:0:a",)),
        )
        result = _check_storage_coverage(self._score_with_coverage(tasks).validate())

        assert result.eligible
        assert not result.has_warnings()

    def test_partial_coverage_below_threshold_warns(self) -> None:
        from sonata.eligibility import _check_storage_coverage

        tasks = (
            Task(task_id=0, func_id=0, core_type="aiv",
                 args=("a", "b", "c"), arg_directions=("input", "output", "output"),
                 arg_storage_keys=("param:0:a", None, None)),
        )
        result = _check_storage_coverage(self._score_with_coverage(tasks).validate())

        assert result.eligible
        assert result.has_warnings()
        assert result.reason_details[0].code == "storage_coverage_below_threshold"
        assert "below threshold" in result.reason_details[0].message

    def test_zero_coverage_with_unknown_args_warns(self) -> None:
        from sonata.eligibility import _check_storage_coverage

        tasks = (
            Task(task_id=0, func_id=0, core_type="aiv",
                 args=("a",), arg_directions=("scalar",), arg_storage_keys=("param:0:a",)),
            Task(task_id=1, func_id=1, core_type="aiv",
                 args=("b",), arg_directions=("input",), arg_storage_keys=(None,)),
        )
        result = _check_storage_coverage(self._score_with_coverage(tasks).validate())

        assert result.eligible
        assert result.has_warnings()
        assert "below threshold" in result.reason_details[0].message

    def test_no_memory_args_eligible(self) -> None:
        from sonata.eligibility import _check_storage_coverage

        tasks = (
            Task(task_id=0, func_id=0, core_type="aiv",
                 args=("s",), arg_directions=("scalar",), arg_storage_keys=(None,)),
        )
        result = _check_storage_coverage(self._score_with_coverage(tasks).validate())

        assert result.eligible
        assert not result.has_warnings()

    def test_no_params_skips_coverage_check(self) -> None:
        from sonata.eligibility import _check_storage_coverage
        from sonata.score import Score, RuntimeTarget

        # No metadata at all — simulates a function with no params
        score = Score(name="no_params", runtime_target=RuntimeTarget(),
                      tasks=(Task(task_id=0, func_id=0, core_type="aiv",
                                  args=("x",), arg_directions=("input",), arg_storage_keys=(None,)),))
        result = _check_storage_coverage(score.validate())

        assert result.eligible
        assert not result.has_warnings()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
