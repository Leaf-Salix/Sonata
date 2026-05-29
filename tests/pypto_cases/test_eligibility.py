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

import pypto.language as pl
import pytest
from pypto import passes
from pypto.pypto_core import passes as _core_passes
from sonata import DEPENDENCY_POLICY_DATAFLOW_V0, RuntimeTarget, check_static_eligibility


@dataclass
class Function:
    name: str
    body: Any


@dataclass
class EvalStmt:
    expr: Any


@dataclass
class AssignStmt:
    var: Any
    value: Any


@dataclass
class Call:
    op: str
    args: tuple[Any, ...] = ()
    arg_directions: tuple[Any, ...] = ()


@dataclass(frozen=True)
class Var:
    name_hint: str


@dataclass
class ForStmt:
    body: Any


@dataclass
class IfStmt:
    then_body: Any
    else_body: Any | None = None


@dataclass
class RuntimeScopeStmt:
    body: Any


def test_static_eligibility_accepts_simple_straight_line_function() -> None:
    func = Function(name="main", body=(EvalStmt(Call("kernel.add")),))

    result = check_static_eligibility(
        func,
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_main_graph"),
    )

    assert result.eligible
    assert result.score is not None
    assert result.score.name == "main"
    assert result.score.runtime_target.runtime == "host_build_graph"
    assert [(task.task_id, task.func_id, task.core_type, task.name) for task in result.score.tasks] == [
        (0, 0, "mixed", "kernel.add")
    ]


def test_static_eligibility_extracts_straight_line_tasks_in_order() -> None:
    func = Function(
        name="main",
        body=(
            EvalStmt(Call("kernel.load", args=("x",))),
            EvalStmt(Call("kernel.compute", args=("x", "tmp"))),
            EvalStmt(Call("kernel.compute", args=("tmp", "out"))),
        ),
    )

    result = check_static_eligibility(func)

    assert result.eligible
    assert result.score is not None
    assert result.score.tasks[0].name == "kernel.load"
    assert result.score.tasks[0].func_id == 0
    assert result.score.tasks[0].args == ("x",)
    assert result.score.tasks[1].name == "kernel.compute"
    assert result.score.tasks[1].func_id == 1
    assert result.score.tasks[1].args == ("x", "tmp")
    assert result.score.tasks[2].name == "kernel.compute"
    assert result.score.tasks[2].func_id == 1
    assert result.score.tasks[2].args == ("tmp", "out")
    assert [(dep.producer, dep.consumer) for dep in result.score.dependencies] == [(0, 1), (1, 2)]
    assert result.score.metadata["dependency_policy"] == "sequential_v0"


def test_static_eligibility_can_use_dataflow_dependency_policy() -> None:
    func = Function(
        name="main",
        body=(
            EvalStmt(Call("kernel.read_x", args=("x",), arg_directions=("Input",))),
            EvalStmt(Call("kernel.read_y", args=("y",), arg_directions=("Input",))),
            EvalStmt(
                Call(
                    "kernel.write_tmp",
                    args=("x", "tmp"),
                    arg_directions=("Input", "OutputExisting"),
                )
            ),
            EvalStmt(Call("kernel.read_tmp", args=("tmp",), arg_directions=("Input",))),
        ),
    )

    result = check_static_eligibility(func, dependency_policy=DEPENDENCY_POLICY_DATAFLOW_V0)

    assert result.eligible
    assert result.score is not None
    assert result.score.metadata["dependency_policy"] == "dataflow_v0"
    assert [(dep.producer, dep.consumer) for dep in result.score.dependencies] == [(2, 3)]


def test_static_eligibility_derives_structural_storage_keys() -> None:
    x = Var("x")
    local = Var("local")
    func = Function(
        name="main",
        body=(
            AssignStmt(local, Call("tensor.create")),
            AssignStmt(
                local,
                Call("kernel.write", args=(x, local), arg_directions=("Input", "OutputExisting")),
            ),
            EvalStmt(Call("kernel.read", args=(local,), arg_directions=("Input",))),
        ),
    )
    func.params = (x,)

    result = check_static_eligibility(func, dependency_policy=DEPENDENCY_POLICY_DATAFLOW_V0)

    assert result.eligible
    assert result.score is not None
    assert result.score.metadata["dependency_policy"] == "dataflow_v0"
    assert result.score.tasks[0].arg_storage_keys[0] == "param:x"
    assert result.score.tasks[0].arg_storage_keys[1] == "alloc:local"
    assert result.score.tasks[1].arg_storage_keys == ("alloc:local",)
    assert [(dep.producer, dep.consumer) for dep in result.score.dependencies] == [(0, 1)]
    assert result.score.metadata["storage_key_coverage"] == {"known": 3, "unknown": 0, "total": 3}
    assert result.score.metadata["memory_storage_key_coverage"] == {"known": 3, "unknown": 0, "total": 3}


def test_static_eligibility_records_unknown_memory_storage_metadata() -> None:
    func = Function(
        name="main",
        body=(
            EvalStmt(
                Call(
                    "kernel.write",
                    args=("x", "tmp", 1),
                    arg_directions=("Input", "OutputExisting", "Scalar"),
                )
            ),
        ),
    )

    result = check_static_eligibility(func, dependency_policy=DEPENDENCY_POLICY_DATAFLOW_V0)

    assert result.eligible
    assert result.score is not None
    assert result.score.metadata["storage_key_coverage"] == {"known": 0, "unknown": 3, "total": 3}
    assert result.score.metadata["memory_storage_key_coverage"] == {"known": 0, "unknown": 2, "total": 2}
    assert result.score.metadata["unknown_memory_storage_args"] == (
        {
            "task_id": 0,
            "task_name": "kernel.write",
            "arg_index": 0,
            "arg": "x",
            "storage_key": None,
        },
        {
            "task_id": 0,
            "task_name": "kernel.write",
            "arg_index": 1,
            "arg": "tmp",
            "storage_key": None,
        },
    )


def test_static_eligibility_records_nodep_storage_metadata_without_edges() -> None:
    func = Function(
        name="main",
        body=(
            EvalStmt(Call("kernel.first", args=("x",), arg_directions=("NoDep",))),
            EvalStmt(Call("kernel.second", args=("x",), arg_directions=("Input",))),
        ),
    )

    result = check_static_eligibility(func, dependency_policy=DEPENDENCY_POLICY_DATAFLOW_V0)

    assert result.eligible
    assert result.score is not None
    assert result.score.dependencies == ()
    assert result.score.metadata["nodep_args"] == (
        {
            "task_id": 0,
            "task_name": "kernel.first",
            "arg_index": 0,
            "arg": "x",
            "storage_key": None,
        },
    )


def test_static_eligibility_records_real_pypto_no_dep_metadata() -> None:
    program = pl.parse_program(
        """
@pl.program
class P:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64], pl.FP32],
        shared: pl.Tensor[[64], pl.FP32],
        out: pl.Out[pl.Tensor[[64], pl.FP32]],
    ) -> pl.Tensor[[64], pl.FP32]:
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[64], pl.FP32],
        shared: pl.Tensor[[64], pl.FP32],
    ) -> pl.Tensor[[64], pl.FP32]:
        local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
        local = self.kernel(x, pl.no_dep(shared), local)
        return local
"""
    )

    instruments = [_core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)]
    with _core_passes.PassContext(instruments):
        program = passes.derive_call_directions()(program)
    result = check_static_eligibility(program, dependency_policy=DEPENDENCY_POLICY_DATAFLOW_V0)

    assert result.eligible
    assert result.score is not None
    assert result.score.dependencies == ()
    assert result.score.tasks[0].arg_directions == ("Input", "NoDep", "OutputExisting")
    assert result.score.tasks[0].arg_storage_keys[1].startswith("param:")
    assert result.score.metadata["nodep_args"] == (
        {
            "task_id": 0,
            "task_name": "kernel",
            "arg_index": 1,
            "arg": "shared",
            "storage_key": result.score.tasks[0].arg_storage_keys[1],
        },
    )


def test_static_eligibility_falls_back_when_dataflow_direction_data_is_missing() -> None:
    func = Function(
        name="main",
        body=(EvalStmt(Call("kernel.a")), EvalStmt(Call("kernel.b"))),
    )

    result = check_static_eligibility(func, dependency_policy=DEPENDENCY_POLICY_DATAFLOW_V0)

    assert result.eligible
    assert result.score is not None
    assert [(dep.producer, dep.consumer) for dep in result.score.dependencies] == [(0, 1)]
    assert result.score.metadata["dependency_policy"] == "sequential_v0"
    assert result.score.metadata["requested_dependency_policy"] == "dataflow_v0"


def test_static_eligibility_rejects_control_flow() -> None:
    func = Function(name="main", body=(ForStmt(body=(EvalStmt(Call("kernel.add")),)), IfStmt(then_body=())))

    result = check_static_eligibility(func)

    assert not result.eligible
    assert result.reasons == (
        "ForStmt is not supported by initial Sonata eligibility",
        "IfStmt is not supported by initial Sonata eligibility",
    )


def test_static_eligibility_rejects_tensor_read() -> None:
    func = Function(name="main", body=(EvalStmt(Call("tensor.read")),))

    result = check_static_eligibility(func)

    assert not result.eligible
    assert result.reasons == ("tensor.read calls are not supported by initial Sonata eligibility",)


def test_static_eligibility_rejects_runtime_scope() -> None:
    func = Function(name="main", body=RuntimeScopeStmt(body=(EvalStmt(Call("kernel.add")),)))

    result = check_static_eligibility(func)

    assert not result.eligible
    assert result.reasons == ("RuntimeScopeStmt is not supported by initial Sonata eligibility",)


def test_static_eligibility_rejects_unsupported_root() -> None:
    result = check_static_eligibility(Call("kernel.add"))

    assert not result.eligible
    assert result.reasons == ("unsupported root for Sonata eligibility: Call",)


def test_static_eligibility_accepts_real_pypto_function() -> None:
    program = pl.parse_program(
        """
@pl.program
class P:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x
"""
    )

    result = check_static_eligibility(program.get_function("main"))

    assert result.eligible
    assert result.score is not None
    assert result.score.name == "main"
    assert result.score.tasks == ()


def test_static_eligibility_extracts_real_pypto_call_task() -> None:
    program = pl.parse_program(
        """
@pl.program
class P:
    @pl.function(type=pl.FunctionType.InCore)
    def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        a = self.k1(x)
        return a
"""
    )

    result = check_static_eligibility(program.get_function("main"))

    assert result.eligible
    assert result.score is not None
    assert result.score.tasks[0].task_id == 0
    assert result.score.tasks[0].func_id == 0
    assert result.score.tasks[0].name == "k1"
    assert result.score.tasks[0].args == ("x",)


def test_static_eligibility_skips_builtin_tensor_create_tasks() -> None:
    program = pl.parse_program(
        """
@pl.program
class P:
    @pl.function(type=pl.FunctionType.InCore)
    def k1(
        self,
        x: pl.Tensor[[64], pl.FP32],
        out: pl.Out[pl.Tensor[[64], pl.FP32]],
    ) -> pl.Tensor[[64], pl.FP32]:
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
        a = self.k1(x, local)
        return a
"""
    )

    result = check_static_eligibility(program)

    assert result.eligible
    assert result.score is not None
    assert [task.name for task in result.score.tasks] == ["k1"]


def test_static_eligibility_uses_real_pypto_arg_directions_for_dataflow() -> None:
    program = pl.parse_program(
        """
@pl.program
class P:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64], pl.FP32],
        out: pl.Out[pl.Tensor[[64], pl.FP32]],
    ) -> pl.Tensor[[64], pl.FP32]:
        t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
        ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
        return ret

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
        local = self.kernel(x, local)
        local = self.kernel(x, local)
        return local
"""
    )

    program = passes.derive_call_directions()(program)
    result = check_static_eligibility(program, dependency_policy=DEPENDENCY_POLICY_DATAFLOW_V0)

    assert result.eligible
    assert result.score is not None
    assert result.score.metadata["dependency_policy"] == "dataflow_v0"
    assert [(task.name, task.args, task.arg_directions) for task in result.score.tasks] == [
        ("kernel", ("x", "local"), ("Input", "OutputExisting")),
        ("kernel", ("x", "local"), ("Input", "InOut")),
    ]
    assert result.score.tasks[0].arg_storage_keys[0].startswith("param:")
    assert result.score.tasks[0].arg_storage_keys[1].startswith("alloc:")
    assert result.score.tasks[0].arg_storage_keys == result.score.tasks[1].arg_storage_keys
    assert [(dep.producer, dep.consumer) for dep in result.score.dependencies] == [(0, 1)]


def test_static_eligibility_propagates_real_tuple_return_storage_keys() -> None:
    program = pl.parse_program(
        """
@pl.program
class P:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64], pl.FP32],
        out0: pl.Out[pl.Tensor[[64], pl.FP32]],
        out1: pl.Out[pl.Tensor[[64], pl.FP32]],
    ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
        return out0, out1

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[64], pl.FP32],
    ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
        local0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
        local1: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
        a, b = self.kernel(x, local0, local1)
        used_a: pl.Tensor[[64], pl.FP32] = self.consume(a)
        used_b: pl.Tensor[[64], pl.FP32] = self.consume(b)
        return used_a, used_b

    @pl.function(type=pl.FunctionType.InCore)
    def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x
"""
    )

    program = passes.derive_call_directions()(program)
    result = check_static_eligibility(program, dependency_policy=DEPENDENCY_POLICY_DATAFLOW_V0)

    assert result.eligible
    assert result.score is not None
    assert [task.name for task in result.score.tasks] == ["kernel", "consume", "consume"]
    kernel_task, consume_a, consume_b = result.score.tasks
    assert kernel_task.arg_storage_keys[1].startswith("alloc:")
    assert kernel_task.arg_storage_keys[2].startswith("alloc:")
    assert consume_a.arg_storage_keys == (kernel_task.arg_storage_keys[1],)
    assert consume_b.arg_storage_keys == (kernel_task.arg_storage_keys[2],)
    assert [(dep.producer, dep.consumer) for dep in result.score.dependencies] == [(0, 1), (0, 2)]


def test_static_eligibility_infers_real_pypto_core_types_from_program() -> None:
    program = pl.parse_program(
        """
@pl.program
class P:
    @pl.function(type=pl.FunctionType.AIC)
    def cube(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    @pl.function(type=pl.FunctionType.AIV)
    def vector(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    @pl.function(type=pl.FunctionType.Group)
    def group(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        a = self.cube(x)
        b = self.vector(a)
        c = self.group(b)
        return c
"""
    )

    result = check_static_eligibility(program)

    assert result.eligible
    assert result.score is not None
    assert [(task.name, task.core_type) for task in result.score.tasks] == [
        ("cube", "aic"),
        ("vector", "aiv"),
        ("group", "mixed"),
    ]
    assert [(dep.producer, dep.consumer) for dep in result.score.dependencies] == [(0, 1), (1, 2)]


def test_static_eligibility_extracts_tasks_only_from_orchestration_functions() -> None:
    program = pl.parse_program(
        """
@pl.program
class P:
    @pl.function(type=pl.FunctionType.AIC)
    def leaf(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    @pl.function(type=pl.FunctionType.AIC)
    def cube(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y = self.leaf(x)
        return y

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        a = self.cube(x)
        return a
"""
    )

    result = check_static_eligibility(program)

    assert result.eligible
    assert result.score is not None
    assert [task.name for task in result.score.tasks] == ["cube"]


def test_static_eligibility_can_select_program_entry_function() -> None:
    program = pl.parse_program(
        """
@pl.program
class P:
    @pl.function(type=pl.FunctionType.AIC)
    def cube(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    @pl.function(type=pl.FunctionType.AIV)
    def vector(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    @pl.function(type=pl.FunctionType.Orchestration)
    def main_a(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        a = self.cube(x)
        return a

    @pl.function(type=pl.FunctionType.Orchestration)
    def main_b(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        b = self.vector(x)
        return b
"""
    )

    result = check_static_eligibility(program, entry_name="main_b")

    assert result.eligible
    assert result.score is not None
    assert [task.name for task in result.score.tasks] == ["vector"]
    assert result.score.dependencies == ()
    assert result.score.metadata["entry_name"] == "main_b"


def test_static_eligibility_rejects_non_orchestration_entry_function() -> None:
    program = pl.parse_program(
        """
@pl.program
class P:
    @pl.function(type=pl.FunctionType.AIC)
    def cube(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        a = self.cube(x)
        return a
"""
    )

    result = check_static_eligibility(program, entry_name="cube")

    assert not result.eligible
    assert result.reasons == ("entry function is not an orchestration function: cube",)


def test_static_eligibility_marks_default_multi_entry_program() -> None:
    program = pl.parse_program(
        """
@pl.program
class P:
    @pl.function(type=pl.FunctionType.AIC)
    def cube(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    @pl.function(type=pl.FunctionType.Orchestration)
    def main_a(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        a = self.cube(x)
        return a

    @pl.function(type=pl.FunctionType.Orchestration)
    def main_b(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        b = self.cube(x)
        return b
"""
    )

    result = check_static_eligibility(program)

    assert result.eligible
    assert result.score is not None
    assert result.score.metadata["entry_policy"] == "all_orchestration"
    assert set(result.score.metadata["entry_names"]) == {"main_a", "main_b"}


def test_static_eligibility_rejects_real_pypto_program_with_loop() -> None:
    program = pl.parse_program(
        """
@pl.program
class P:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for i in pl.range(4):
            x = pl.add(x, 1.0)
        return x
"""
    )

    result = check_static_eligibility(program)

    assert not result.eligible
    assert result.reasons == ("ForStmt is not supported by initial Sonata eligibility",)


def test_static_eligibility_rejects_real_pypto_program_with_runtime_scope() -> None:
    program = pl.parse_program(
        """
@pl.program
class P:
    @pl.function(type=pl.FunctionType.InCore)
    def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.auto_scope():
            a = self.k1(x)
        return a
"""
    )

    result = check_static_eligibility(program)

    assert not result.eligible
    assert result.reasons == ("RuntimeScopeStmt is not supported by initial Sonata eligibility",)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
