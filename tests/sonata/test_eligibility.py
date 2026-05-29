from dataclasses import dataclass
from typing import Any

from sonata import DEPENDENCY_POLICY_DATAFLOW_V0, RuntimeTarget, check_static_eligibility


@dataclass
class Function:
    name: str
    body: Any


@dataclass
class NamelessFunction:
    body: Any


@dataclass
class Program:
    functions: dict[str, Any]
    name: str = "P"


@dataclass(frozen=True)
class FuncType:
    name: str


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
    type: Any | None = None


@dataclass(frozen=True)
class TensorType:
    shape: tuple[Any, ...]


@dataclass(frozen=True)
class ShapeUnderscoreType:
    shape_: tuple[Any, ...]


@dataclass(frozen=True)
class DimsType:
    dims: tuple[Any, ...]


@dataclass(frozen=True)
class DimsUnderscoreType:
    dims_: tuple[Any, ...]


@dataclass(frozen=True)
class ConstInt:
    value: int


@dataclass(frozen=True)
class TypeUnderscoreVar:
    name_hint: str
    type_: Any | None = None


@dataclass(frozen=True)
class TensorTypeAttrVar:
    name_hint: str
    tensor_type: Any | None = None


@dataclass(frozen=True)
class AnonymousVar:
    type: Any | None = None


@dataclass
class ForStmt:
    body: Any


@dataclass
class IfStmt:
    then_body: Any
    else_body: Any | None = None


@dataclass
class WhileStmt:
    body: Any


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


def test_static_eligibility_defaults_to_host_build_graph_runtime_target() -> None:
    func = Function(name="main", body=(EvalStmt(Call("kernel.add")),))

    result = check_static_eligibility(func)

    assert result.eligible
    assert result.score is not None
    assert result.score.runtime_target.runtime == "host_build_graph"
    assert result.score.runtime_target.function_name == "build_main_graph"


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


def test_static_eligibility_extracts_static_shape_assumptions() -> None:
    x = Var("x", TensorType((ConstInt(64), 32)))
    dyn = Var("dyn", TensorType((ConstInt(64), "n")))
    no_shape = Var("no_shape")
    func = Function(name="main", body=(EvalStmt(Call("kernel.add", args=(x, dyn, no_shape))),))
    func.params = (x, dyn, no_shape)

    result = check_static_eligibility(func)

    assert result.eligible
    assert result.score is not None
    assert [(shape.symbol, shape.dims) for shape in result.score.shape_assumptions] == [("x", (64, 32))]


def test_static_eligibility_extracts_alternate_shape_field_names() -> None:
    shape_under = TypeUnderscoreVar("shape_under", ShapeUnderscoreType((8, ConstInt(16))))
    dims = TensorTypeAttrVar("dims", DimsType((32,)))
    dims_under = Var("dims_under", DimsUnderscoreType((64,)))
    func = Function(
        name="main",
        body=(EvalStmt(Call("kernel.add", args=(shape_under, dims, dims_under))),),
    )
    func.params = (shape_under, dims, dims_under)

    result = check_static_eligibility(func)

    assert result.eligible
    assert result.score is not None
    assert [(shape.symbol, shape.dims) for shape in result.score.shape_assumptions] == [
        ("shape_under", (8, 16)),
        ("dims", (32,)),
        ("dims_under", (64,)),
    ]


def test_static_eligibility_qualifies_shape_assumptions_for_multiple_entries() -> None:
    x_a = Var("x", TensorType((64,)))
    x_b = Var("x", TensorType((128,)))
    main_a = Function(name="main_a", body=(EvalStmt(Call("kernel.a", args=(x_a,))),))
    main_b = Function(name="main_b", body=(EvalStmt(Call("kernel.b", args=(x_b,))),))
    main_a.params = (x_a,)
    main_b.params = (x_b,)
    main_a.func_type = FuncType("Orchestration")
    main_b.func_type = FuncType("Orchestration")

    result = check_static_eligibility(Program(functions={"main_a": main_a, "main_b": main_b}))

    assert result.eligible
    assert result.score is not None
    assert [(shape.symbol, shape.dims) for shape in result.score.shape_assumptions] == [
        ("main_a.x", (64,)),
        ("main_b.x", (128,)),
    ]


def test_static_eligibility_rejects_ambiguous_unqualified_shape_assumptions() -> None:
    x_a = Var("x", TensorType((64,)))
    x_b = Var("x", TensorType((128,)))
    main_a = NamelessFunction(body=(EvalStmt(Call("kernel.a", args=(x_a,))),))
    main_b = NamelessFunction(body=(EvalStmt(Call("kernel.b", args=(x_b,))),))
    main_a.params = (x_a,)
    main_b.params = (x_b,)
    main_a.func_type = FuncType("Orchestration")
    main_b.func_type = FuncType("Orchestration")

    result = check_static_eligibility(Program(functions={"main_a": main_a, "main_b": main_b}))

    assert not result.eligible
    assert result.reasons == ("shape assumption symbol must be unique: x",)


def test_static_eligibility_selects_explicit_orchestration_entry_name() -> None:
    x_a = Var("x_a", TensorType((64,)))
    x_b = Var("x_b", TensorType((128,)))
    main_a = Function(name="main_a", body=(EvalStmt(Call("kernel.a", args=(x_a,))),))
    main_b = Function(name="main_b", body=(EvalStmt(Call("kernel.b", args=(x_b,))),))
    main_a.params = (x_a,)
    main_b.params = (x_b,)
    main_a.func_type = FuncType("Orchestration")
    main_b.func_type = FuncType("Orchestration")

    result = check_static_eligibility(
        Program(functions={"main_a": main_a, "main_b": main_b}),
        entry_name="main_b",
    )

    assert result.eligible
    assert result.score is not None
    assert [task.name for task in result.score.tasks] == ["kernel.b"]
    assert [(shape.symbol, shape.dims) for shape in result.score.shape_assumptions] == [("x_b", (128,))]


def test_static_eligibility_rejects_explicit_non_orchestration_entry_name() -> None:
    helper = Function(name="helper", body=(EvalStmt(Call("kernel.helper")),))
    helper.func_type = FuncType("AIC")

    result = check_static_eligibility(Program(functions={"helper": helper}), entry_name="helper")

    assert not result.eligible
    assert result.reasons == ("entry function is not an orchestration function: helper",)


def test_static_eligibility_skips_shape_assumptions_without_stable_symbol() -> None:
    unnamed = AnonymousVar(TensorType((16,)))
    func = Function(name="main", body=(EvalStmt(Call("kernel.add", args=(unnamed,))),))
    func.params = (unnamed,)

    result = check_static_eligibility(func)

    assert result.eligible
    assert result.score is not None
    assert result.score.shape_assumptions == ()


def test_static_eligibility_skips_negative_shape_sentinel() -> None:
    x = Var("x", TensorType((ConstInt(-1), ConstInt(32))))
    func = Function(name="main", body=(EvalStmt(Call("kernel.add", args=(x,))),))
    func.params = (x,)

    result = check_static_eligibility(func)

    assert result.eligible
    assert result.score is not None
    assert result.score.shape_assumptions == ()


def test_static_eligibility_skips_zero_shape_sentinel() -> None:
    x = Var("x", TensorType((ConstInt(0), ConstInt(32))))
    func = Function(name="main", body=(EvalStmt(Call("kernel.add", args=(x,))),))
    func.params = (x,)

    result = check_static_eligibility(func)

    assert result.eligible
    assert result.score is not None
    assert result.score.shape_assumptions == ()


def test_static_eligibility_skips_bool_shape_dimensions() -> None:
    flag_dim = Var("flag_dim", TensorType((True, 32)))
    const_flag_dim = Var("const_flag_dim", TensorType((ConstInt(True), 32)))
    valid = Var("valid", TensorType((16, 32)))
    func = Function(
        name="main",
        body=(EvalStmt(Call("kernel.add", args=(flag_dim, const_flag_dim, valid))),),
    )
    func.params = (flag_dim, const_flag_dim, valid)

    result = check_static_eligibility(func)

    assert result.eligible
    assert result.score is not None
    assert [(shape.symbol, shape.dims) for shape in result.score.shape_assumptions] == [
        ("valid", (16, 32))
    ]


def test_static_eligibility_rejects_control_flow() -> None:
    func = Function(name="main", body=(ForStmt(body=(EvalStmt(Call("kernel.add")),)), IfStmt(then_body=())))

    result = check_static_eligibility(func)

    assert not result.eligible
    assert result.reasons == (
        "ForStmt is not supported by initial Sonata eligibility",
        "IfStmt is not supported by initial Sonata eligibility",
    )


def test_static_eligibility_dedupes_duplicate_fallback_reasons() -> None:
    func = Function(
        name="main",
        body=(
            ForStmt(body=(EvalStmt(Call("kernel.add")),)),
            ForStmt(body=(EvalStmt(Call("kernel.mul")),)),
        ),
    )

    result = check_static_eligibility(func)

    assert not result.eligible
    assert result.reasons == ("ForStmt is not supported by initial Sonata eligibility",)


def test_static_eligibility_rejects_while_control_flow() -> None:
    func = Function(name="main", body=(WhileStmt(body=(EvalStmt(Call("kernel.add")),)),))

    result = check_static_eligibility(func)

    assert not result.eligible
    assert result.reasons == ("WhileStmt is not supported by initial Sonata eligibility",)


def test_static_eligibility_rejects_unsupported_root() -> None:
    result = check_static_eligibility(object())

    assert not result.eligible
    assert result.reasons == ("unsupported root for Sonata eligibility: object",)


def test_static_eligibility_rejects_runtime_scope() -> None:
    func = Function(name="main", body=RuntimeScopeStmt(body=(EvalStmt(Call("kernel.add")),)))

    result = check_static_eligibility(func)

    assert not result.eligible
    assert result.reasons == ("RuntimeScopeStmt is not supported by initial Sonata eligibility",)


def test_static_eligibility_rejects_tensor_read() -> None:
    func = Function(name="main", body=(EvalStmt(Call("tensor.read")),))

    result = check_static_eligibility(func)

    assert not result.eligible
    assert result.reasons == ("tensor.read calls are not supported by initial Sonata eligibility",)
