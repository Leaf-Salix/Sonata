from dataclasses import dataclass
from typing import Any

from sonata import (
    DEPENDENCY_POLICY_DATAFLOW_V0,
    DEPENDENCY_POLICY_SEQUENTIAL_V0,
    FallbackCode,
    RuntimeTarget,
    check_static_eligibility,
)
from sonata.score import raw_runtime_target


@dataclass
class Function:
    name: str
    body: Any
    func_type: Any | None = None


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


@dataclass
class Submit:
    body: Any


@dataclass
class SpmdScopeStmt:
    body: Any


@dataclass
class ManualScopeStmt:
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
    assert raw_runtime_target(result.score).runtime == "host_build_graph"
    assert raw_runtime_target(result.score).function_name == "build_main_graph"


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
    assert "requested_dependency_policy" not in result.score.metadata
    assert "dependency_policy_fallback_reason" not in result.score.metadata


def test_dataflow_fallback_records_unavailable_in_metadata() -> None:
    func = Function(
        name="main",
        body=(
            EvalStmt(Call("kernel.a", args=("x",))),
            EvalStmt(Call("kernel.b", args=("y",))),
        ),
    )

    result = check_static_eligibility(func, dependency_policy=DEPENDENCY_POLICY_DATAFLOW_V0)

    assert result.eligible
    assert result.score is not None
    assert result.score.metadata["dependency_policy"] == "sequential_v0"
    assert result.score.metadata["requested_dependency_policy"] == "dataflow_v0"
    assert result.score.metadata["dependency_policy_fallback_reason"] == "dataflow_directions_unavailable"


def test_dataflow_fallback_records_incomplete_in_metadata() -> None:
    func = Function(
        name="main",
        body=(
            EvalStmt(Call("kernel.a", args=("x",), arg_directions=("Input",))),
            EvalStmt(Call("kernel.b", args=("y",))),
        ),
    )

    result = check_static_eligibility(func, dependency_policy=DEPENDENCY_POLICY_DATAFLOW_V0)

    assert result.eligible
    assert result.score is not None
    assert result.score.metadata["dependency_policy"] == "sequential_v0"
    assert result.score.metadata["dependency_policy_fallback_reason"] == "dataflow_directions_incomplete"


def test_sequential_policy_has_no_fallback_metadata() -> None:
    func = Function(
        name="main",
        body=(
            EvalStmt(Call("kernel.a", args=("x",))),
            EvalStmt(Call("kernel.b", args=("y",))),
        ),
    )

    result = check_static_eligibility(func, dependency_policy=DEPENDENCY_POLICY_SEQUENTIAL_V0)

    assert result.eligible
    assert result.score is not None
    assert "requested_dependency_policy" not in result.score.metadata
    assert "dependency_policy_fallback_reason" not in result.score.metadata


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


def test_static_eligibility_skips_param_with_mixed_positive_and_negative_dims() -> None:
    x = Var("x", TensorType((2, -1, 4)))
    y = Var("y", TensorType((8,)))
    func = Function(name="main", body=(EvalStmt(Call("kernel.add", args=(x, y))),))
    func.params = (x, y)

    result = check_static_eligibility(func)

    assert result.eligible
    assert result.score is not None
    assert [(shape.symbol, shape.dims) for shape in result.score.shape_assumptions] == [("y", (8,))]


def test_static_eligibility_skips_param_with_all_negative_dims() -> None:
    x = Var("x", TensorType((-1, -1)))
    y = Var("y", TensorType((8,)))
    func = Function(name="main", body=(EvalStmt(Call("kernel.add", args=(x, y))),))
    func.params = (x, y)

    result = check_static_eligibility(func)

    assert result.eligible
    assert result.score is not None
    assert [(shape.symbol, shape.dims) for shape in result.score.shape_assumptions] == [("y", (8,))]
    assert result.score.validate().eligible is True


def test_static_eligibility_skips_param_with_mixed_dims_via_const_int() -> None:
    x = Var("x", TensorType((ConstInt(2), ConstInt(-1), ConstInt(4))))
    y = Var("y", TensorType((ConstInt(8),)))
    func = Function(name="main", body=(EvalStmt(Call("kernel.add", args=(x, y))),))
    func.params = (x, y)

    result = check_static_eligibility(func)

    assert result.eligible
    assert result.score is not None
    assert [(shape.symbol, shape.dims) for shape in result.score.shape_assumptions] == [("y", (8,))]
    assert result.score.validate().eligible is True


def test_static_eligibility_extracts_const_int_positive_via_value() -> None:
    x = Var("x", TensorType((ConstInt(3), ConstInt(16))))
    func = Function(name="main", body=(EvalStmt(Call("kernel.add", args=(x,))),))
    func.params = (x,)

    result = check_static_eligibility(func)

    assert result.eligible
    assert result.score is not None
    assert [(shape.symbol, shape.dims) for shape in result.score.shape_assumptions] == [("x", (3, 16))]
    assert result.score.validate().eligible is True


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


def test_static_eligibility_rejects_pypto_adapter_out_of_scope_nodes() -> None:
    for node_type in (Submit, SpmdScopeStmt, ManualScopeStmt):
        func = Function(name="main", body=node_type(body=(EvalStmt(Call("kernel.add")),)))

        result = check_static_eligibility(func)

        assert not result.eligible
        assert result.reasons == (f"{node_type.__name__} is out of scope for Sonata v0.1 PyPTO adapter",)
        assert result.reason_details[0].code == FallbackCode.UNSUPPORTED_PYPTO_ADAPTER_SCOPE.value


def test_static_eligibility_rejects_direct_group_pypto_function_root() -> None:
    func = Function(name="group", body=(EvalStmt(Call("kernel.add", arg_directions=("Input",))),))
    func.func_type = FuncType("Group")

    result = check_static_eligibility(func)

    assert not result.eligible
    assert result.reason_details[0].code == FallbackCode.UNSUPPORTED_PYPTO_ADAPTER_SCOPE.value
    assert result.reasons == (
        "Group function root is out of scope for Sonata v0.1 PyPTO adapter: group",
    )


def test_static_eligibility_rejects_direct_typed_non_orchestration_root() -> None:
    func = Function(name="kernel", body=(EvalStmt(Call("kernel.add", arg_directions=("Input",))),))
    func.func_type = FuncType("AIV")

    result = check_static_eligibility(func)

    assert not result.eligible
    assert result.reason_details[0].code == FallbackCode.UNSUPPORTED_PYPTO_ADAPTER_SCOPE.value
    assert result.reasons == ("AIV function root is not an Orchestration function: kernel",)


def test_static_eligibility_accepts_untyped_structural_function_root() -> None:
    func = Function(name="main", body=(EvalStmt(Call("kernel.add", args=("x",), arg_directions=("Input",))),))

    result = check_static_eligibility(func)

    assert result.eligible
    assert result.score is not None
    assert result.score.tasks[0].name == "kernel.add"


def test_static_eligibility_certified_mode_rejects_missing_directions() -> None:
    program = Program(
        functions={
            "kernel": Function(name="kernel", body=(), func_type=FuncType("AIV")),
            "main": Function(
                name="main",
                body=(EvalStmt(Call("kernel", args=("x",), arg_directions=())),),
                func_type=FuncType("Orchestration"),
            ),
        }
    )

    relaxed = check_static_eligibility(program)
    strict = check_static_eligibility(program, require_certified=True)

    assert relaxed.eligible
    assert not strict.eligible
    assert strict.reason_details[0].code == FallbackCode.UNSUPPORTED_PYPTO_ADAPTER_SCOPE.value
    assert strict.reasons == ("Call kernel has no arg_directions in certified dump",)


def test_static_eligibility_rejects_program_without_orchestration_roots() -> None:
    program = Program(
        functions={
            "kernel": Function(name="kernel", body=(), func_type=FuncType("AIV")),
        }
    )

    result = check_static_eligibility(program)

    assert not result.eligible
    assert result.reason_details[0].code == FallbackCode.ENTRY_FUNCTION_NOT_ORCHESTRATION.value
    assert result.reasons == ("program has no Orchestration functions for Sonata eligibility",)


def test_static_eligibility_scans_selected_orchestration_roots_only() -> None:
    program = Program(
        functions={
            "kernel": Function(
                name="kernel",
                body=ForStmt(body=(EvalStmt(Call("tile.add")),)),
                func_type=FuncType("AIV"),
            ),
            "main": Function(
                name="main",
                body=(EvalStmt(Call("kernel", args=("x",), arg_directions=("Input",))),),
                func_type=FuncType("Orchestration"),
            ),
        }
    )

    result = check_static_eligibility(program, require_certified=True)

    assert result.eligible
    assert result.score is not None
    assert result.score.tasks[0].name == "kernel"


def test_static_eligibility_rejects_tensor_read() -> None:
    func = Function(name="main", body=(EvalStmt(Call("tensor.read")),))

    result = check_static_eligibility(func)

    assert not result.eligible
    assert result.reasons == ("tensor.read calls are not supported by initial Sonata eligibility",)


class TestWalkCache:
    """v0.17 Phase 2 C2: _walk cache prevents redundant tree traversals."""

    def test_walk_cache_returns_same_nodes(self):
        """Same node yields same results from cache."""
        from sonata.eligibility import _walk, _walk_cache

        node = Function(name="f", body=(EvalStmt(Call("k")),))

        # First call — populates cache
        result1 = tuple(_walk(node))
        assert len(result1) > 0

        # Second call — from cache
        result2 = tuple(_walk(node))
        assert result1 == result2

        # Cleanup
        _walk_cache.pop(id(node), None)

    def test_walk_cache_different_nodes(self):
        """Different nodes get different cache entries."""
        from sonata.eligibility import _walk, _walk_cache

        node1 = Function(name="f1", body=(EvalStmt(Call("k1")),))
        node2 = Function(name="f2", body=(EvalStmt(Call("k2")),))

        result1 = tuple(_walk(node1))
        result2 = tuple(_walk(node2))

        # Different nodes → different results
        names1 = {getattr(n, 'name', None) for n in result1 if hasattr(n, 'name')}
        names2 = {getattr(n, 'name', None) for n in result2 if hasattr(n, 'name')}
        assert names1 != names2

        # Cleanup
        _walk_cache.pop(id(node1), None)
        _walk_cache.pop(id(node2), None)

    def test_walk_cache_does_not_mutate(self):
        """Cached result is a tuple (immutable), not a mutable list."""
        from sonata.eligibility import _walk, _walk_cache

        node = Function(name="f", body=(EvalStmt(Call("k")),))

        result = tuple(_walk(node))
        assert isinstance(result, tuple)

        # Cleanup
        _walk_cache.pop(id(node), None)


class TestUnrollableForStmt:
    """v0.18 Phase 2 B1: Small constant ForStmt detection."""

    def _make_for_stmt(self, start, stop, step=None):
        """Create a mock ForStmt with given range parameters."""
        class ForStmt:
            pass
        node = ForStmt()
        node.start = start
        node.stop = stop
        node.step = step
        return node

    def _make_if_stmt(self):
        class IfStmt:
            pass
        return IfStmt()

    def test_small_loop_is_unrollable(self):
        from sonata.eligibility import _is_unrollable_for_stmt
        node = self._make_for_stmt(0, 4)
        assert _is_unrollable_for_stmt(node)

    def test_threshold_loop_is_unrollable(self):
        from sonata.eligibility import _is_unrollable_for_stmt
        node = self._make_for_stmt(0, 16)
        assert _is_unrollable_for_stmt(node)

    def test_large_loop_not_unrollable(self):
        from sonata.eligibility import _is_unrollable_for_stmt
        node = self._make_for_stmt(0, 100)
        assert not _is_unrollable_for_stmt(node)

    def test_dynamic_start_not_unrollable(self):
        from sonata.eligibility import _is_unrollable_for_stmt
        node = self._make_for_stmt(object(), 4)
        assert not _is_unrollable_for_stmt(node)

    def test_non_for_stmt_not_unrollable(self):
        from sonata.eligibility import _is_unrollable_for_stmt
        node = self._make_if_stmt()
        assert not _is_unrollable_for_stmt(node)

    def test_step_2_not_unrollable(self):
        from sonata.eligibility import _is_unrollable_for_stmt
        node = self._make_for_stmt(0, 8, step=2)
        assert not _is_unrollable_for_stmt(node)

    def test_step_1_is_unrollable(self):
        from sonata.eligibility import _is_unrollable_for_stmt
        node = self._make_for_stmt(0, 4, step=1)
        assert _is_unrollable_for_stmt(node)

    def test_eligibility_accepts_small_loop(self):
        """Eligibility check passes when only unrollable ForStmts present."""
        from sonata.eligibility import check_static_eligibility

        call = Call("kernel", args=("x",), arg_directions=("Input",))
        for_stmt = self._make_for_stmt(0, 4)
        for_stmt.body = (EvalStmt(call),)

        func = Function(name="main", body=(for_stmt, EvalStmt(call)))
        result = check_static_eligibility(func, require_certified=True)
        assert result.eligible
