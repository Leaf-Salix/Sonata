from dataclasses import dataclass
from typing import Any

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
    assert [(task.task_id, task.func_id, task.core_type, task.name) for task in result.score.tasks] == [
        (0, 0, "mixed", "kernel.add")
    ]


def test_static_eligibility_can_use_dataflow_dependency_policy() -> None:
    func = Function(
        name="main",
        body=(
            EvalStmt(Call("kernel.read_x", args=("x",), arg_directions=("Input",))),
            EvalStmt(Call("kernel.read_y", args=("y",), arg_directions=("Input",))),
            EvalStmt(Call("kernel.write_tmp", args=("x", "tmp"), arg_directions=("Input", "OutputExisting"))),
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
            AssignStmt(local, Call("kernel.write", args=(x, local), arg_directions=("Input", "OutputExisting"))),
            EvalStmt(Call("kernel.read", args=(local,), arg_directions=("Input",))),
        ),
    )
    func.params = (x,)

    result = check_static_eligibility(func, dependency_policy=DEPENDENCY_POLICY_DATAFLOW_V0)

    assert result.eligible
    assert result.score is not None
    assert result.score.tasks[0].arg_storage_keys == ("param:x", "alloc:local")
    assert result.score.tasks[1].arg_storage_keys == ("alloc:local",)
    assert [(dep.producer, dep.consumer) for dep in result.score.dependencies] == [(0, 1)]


def test_static_eligibility_rejects_control_flow() -> None:
    func = Function(name="main", body=(ForStmt(body=(EvalStmt(Call("kernel.add")),)), IfStmt(then_body=())))

    result = check_static_eligibility(func)

    assert not result.eligible
    assert result.reasons == (
        "ForStmt is not supported by initial Sonata eligibility",
        "IfStmt is not supported by initial Sonata eligibility",
    )


def test_static_eligibility_rejects_runtime_scope() -> None:
    func = Function(name="main", body=RuntimeScopeStmt(body=(EvalStmt(Call("kernel.add")),)))

    result = check_static_eligibility(func)

    assert not result.eligible
    assert result.reasons == ("RuntimeScopeStmt is not supported by initial Sonata eligibility",)
