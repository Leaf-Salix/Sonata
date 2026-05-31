from dataclasses import dataclass
from typing import Any

import pytest
from sonata.pypto_adapter import (
    DEFAULT_CERTIFIED_DUMP,
    NormalizedDependencyFact,
    PostSimplifyPyPTOInputAdapter,
    PyPTOAdapterContractError,
)


@dataclass
class Program:
    functions: dict[str, Any]
    name: str = "P"


@dataclass
class Function:
    name: str
    body: Any
    func_type: Any | None = None
    params: tuple[Any, ...] = ()


@dataclass(frozen=True)
class FuncType:
    name: str


@dataclass
class EvalStmt:
    expr: Any


@dataclass
class Call:
    op: str
    args: tuple[Any, ...] = ()
    arg_directions: tuple[Any, ...] = ()


@dataclass(frozen=True)
class Var:
    name_hint: str


@dataclass
class Submit:
    body: Any


def test_adapter_declares_named_certified_dump_selector() -> None:
    assert DEFAULT_CERTIFIED_DUMP == "after_collect_comm_groups_simplify"
    assert PostSimplifyPyPTOInputAdapter.certified_dump == "after_collect_comm_groups_simplify"


def test_adapter_normalizes_orchestration_calls_without_runtime_artifacts() -> None:
    x = Var("x")
    out = Var("out")
    program = Program(
        functions={
            "kernel": Function(
                name="kernel",
                func_type=FuncType("AIV"),
                body=(),
            ),
            "main": Function(
                name="main",
                func_type=FuncType("Orchestration"),
                params=(x,),
                body=(EvalStmt(Call("kernel", args=(x, out), arg_directions=("Input", "OutputExisting"))),),
            ),
        }
    )

    facts = PostSimplifyPyPTOInputAdapter(program).normalize(require_certified=True)

    assert facts.certified_dump == "after_collect_comm_groups_simplify"
    assert len(facts.functions) == 1
    function = facts.functions[0]
    assert function.name == "main"
    assert function.is_orchestration
    assert function.params == (x,)
    assert len(function.calls) == 1
    call = function.calls[0]
    assert call.callee_name == "kernel"
    assert call.args == (x, out)
    assert call.arg_names == ("x", "out")
    assert call.arg_directions == ("Input", "OutputExisting")
    assert call.arg_storage_keys == ()
    assert call.core_type == "aiv"


def test_adapter_keeps_dependency_fact_extension_fields_available() -> None:
    fact = NormalizedDependencyFact(
        producer=1,
        consumer=2,
        source="explicit",
        hazard="RAW",
        provenance={"arg_index": 0},
    )

    assert fact.producer == 1
    assert fact.consumer == 2
    assert fact.source == "explicit"
    assert fact.hazard == "RAW"
    assert fact.provenance == {"arg_index": 0}


@pytest.mark.parametrize("func_type_name", ["Group", "Spmd"])
def test_adapter_detects_group_or_spmd_callee_as_out_of_scope(func_type_name: str) -> None:
    program = Program(
        functions={
            "unsupported": Function(name="unsupported", func_type=FuncType(func_type_name), body=()),
            "main": Function(
                name="main",
                func_type=FuncType("Orchestration"),
                body=(EvalStmt(Call("unsupported")),),
            ),
        }
    )

    adapter = PostSimplifyPyPTOInputAdapter(program)

    assert adapter.has_unsupported_function_call() == "unsupported"
    with pytest.raises(PyPTOAdapterContractError, match="Group/Spmd callee"):
        adapter.normalize()


def test_adapter_rejects_direct_group_or_spmd_function_root() -> None:
    adapter = PostSimplifyPyPTOInputAdapter(
        Function(
            name="group",
            func_type=FuncType("Group"),
            body=(EvalStmt(Call("kernel", arg_directions=("Input",))),),
        )
    )

    assert adapter.extraction_roots() == ()
    with pytest.raises(PyPTOAdapterContractError, match="Group function root is out of scope"):
        adapter.normalize()


def test_adapter_rejects_direct_typed_non_orchestration_function_root() -> None:
    adapter = PostSimplifyPyPTOInputAdapter(
        Function(
            name="kernel",
            func_type=FuncType("AIV"),
            body=(EvalStmt(Call("kernel", arg_directions=("Input",))),),
        )
    )

    assert adapter.extraction_roots() == ()
    with pytest.raises(PyPTOAdapterContractError, match="AIV function root is not an Orchestration function"):
        adapter.normalize()


def test_certified_adapter_requires_arg_directions_to_match_args() -> None:
    x = Var("x")
    program = Program(
        functions={
            "kernel": Function(name="kernel", func_type=FuncType("AIV"), body=()),
            "main": Function(
                name="main",
                func_type=FuncType("Orchestration"),
                body=(EvalStmt(Call("kernel", args=(x,), arg_directions=())),),
            ),
        }
    )

    with pytest.raises(PyPTOAdapterContractError, match="no arg_directions"):
        PostSimplifyPyPTOInputAdapter(program).normalize(require_certified=True)


def test_adapter_rejects_submit_scope_before_normalizing() -> None:
    program = Program(
        functions={
            "main": Function(
                name="main",
                func_type=FuncType("Orchestration"),
                body=(Submit(body=(EvalStmt(Call("kernel", arg_directions=("Input",))),),),),
            ),
        }
    )

    with pytest.raises(PyPTOAdapterContractError, match="Submit is out of scope"):
        PostSimplifyPyPTOInputAdapter(program).normalize()
