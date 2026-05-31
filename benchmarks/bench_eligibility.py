#!/usr/bin/env python3
"""Benchmark check_static_eligibility with mock IR structures.

Measures eligibility check time for IR-like structures of varying complexity:
  - Different numbers of calls (1, 10, 50, 100, 500)
  - Nested function bodies
  - Mixed eligible/ineligible structures

Uses simple mock objects (Function, EvalStmt, Call dataclasses) that satisfy
the PostSimplifyPyPTOInputAdapter structural interface without depending on
PyPTO C++ bindings.

Usage:
    PYTHONPATH=src python benchmarks/bench_eligibility.py
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from _bench_utils import (
    print_results,
    run_benchmark,
    save_results,
)
from sonata import check_static_eligibility


# ---------------------------------------------------------------------------
# Mock IR objects
# ---------------------------------------------------------------------------
# These mock the structural interface expected by PostSimplifyPyPTOInputAdapter.
# Class names matter (type(node).__name__ is used for kind checks).


@dataclass
class _FuncType:
    """Mock function type with a name attribute."""
    name: str


@dataclass
class MockArg:
    """Mock argument node."""
    name_hint: str = "arg"
    name: str = "arg"


@dataclass
class Call:
    """Mock Call node."""
    op: str = "kernel_launch"
    args: tuple = ()
    arg_directions: tuple = ()


@dataclass
class EvalStmt:
    """Mock EvalStmt node (a simple non-call statement)."""
    value: Any = None


@dataclass
class Function:
    """Mock Function node (used as both root and callee)."""
    name: str = "orchestration_main"
    func_type: _FuncType = field(default_factory=lambda: _FuncType("Orchestration"))
    params: tuple = ()
    body: tuple = ()


@dataclass
class ForStmt:
    """Mock ForStmt node (control flow -- makes eligibility fail)."""
    body: tuple = ()
    condition: Any = None


@dataclass
class IfStmt:
    """Mock IfStmt node (control flow -- makes eligibility fail)."""
    then_body: tuple = ()
    else_body: tuple = ()
    condition: Any = None


@dataclass
class Program:
    """Mock Program node with a functions dict."""
    functions: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mock IR builders
# ---------------------------------------------------------------------------


def build_eligible_function(num_calls: int, *, name: str = "orchestration_main") -> Function:
    """Build a mock Function with ``num_calls`` Call children.

    The function is an Orchestration type with simple EvalStmt and Call
    children. This should pass eligibility checks.
    """
    body_items: list = []
    for i in range(num_calls):
        # Each call has 2 args with directions
        args = (MockArg(name_hint=f"x_{i}"), MockArg(name_hint=f"y_{i}"))
        call = Call(
            op=f"kernel_{i % 8}",
            args=args,
            arg_directions=("in", "out"),
        )
        body_items.append(call)
        # Interleave with some EvalStmts
        if i % 3 == 0:
            body_items.append(EvalStmt(value=f"temp_{i}"))

    return Function(
        name=name,
        func_type=_FuncType("Orchestration"),
        params=(MockArg(name_hint="input_tensor"),),
        body=tuple(body_items),
    )


def build_ineligible_function(num_calls: int) -> Function:
    """Build a mock Function that is NOT eligible (contains ForStmt)."""
    body_items: list = []
    for i in range(num_calls):
        call = Call(
            op=f"kernel_{i % 4}",
            args=(MockArg(name_hint=f"a_{i}"),),
            arg_directions=("in",),
        )
        body_items.append(call)

    # Insert a ForStmt to make it ineligible
    body_items.insert(len(body_items) // 2, ForStmt(body=(EvalStmt(),)))

    return Function(
        name="ineligible_func",
        func_type=_FuncType("Orchestration"),
        params=(),
        body=tuple(body_items),
    )


def build_nested_program(num_functions: int, calls_per_function: int) -> Program:
    """Build a mock Program with multiple orchestration functions."""
    functions = {}
    for f in range(num_functions):
        fname = f"orch_func_{f}"
        functions[fname] = build_eligible_function(calls_per_function, name=fname)
    return Program(functions=functions)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


CALL_COUNTS = [1, 10, 50, 100, 500]


def bench_eligible(num_calls: int) -> dict:
    """Benchmark eligibility check on an eligible function."""
    func = build_eligible_function(num_calls)

    def check():
        return check_static_eligibility(func)

    result = run_benchmark(
        "eligibility_eligible",
        check,
        params={"num_calls": num_calls},
    )
    # Record whether the result was actually eligible
    res = check()
    result["parameters"]["is_eligible"] = res.eligible
    result["parameters"]["num_tasks"] = (
        res.score.task_count() if res.score else 0
    )
    return result


def bench_ineligible(num_calls: int) -> dict:
    """Benchmark eligibility check on an ineligible function (has ForStmt)."""
    func = build_ineligible_function(num_calls)

    def check():
        return check_static_eligibility(func)

    result = run_benchmark(
        "eligibility_ineligible",
        check,
        params={"num_calls": num_calls},
    )
    res = check()
    result["parameters"]["is_eligible"] = res.eligible
    result["parameters"]["num_reasons"] = len(res.reasons)
    return result


def bench_program(num_functions: int, calls_per_function: int) -> dict:
    """Benchmark eligibility check on a multi-function Program."""
    program = build_nested_program(num_functions, calls_per_function)

    def check():
        return check_static_eligibility(program)

    result = run_benchmark(
        "eligibility_program",
        check,
        params={
            "num_functions": num_functions,
            "calls_per_function": calls_per_function,
        },
    )
    res = check()
    result["parameters"]["is_eligible"] = res.eligible
    result["parameters"]["total_tasks"] = (
        res.score.task_count() if res.score else 0
    )
    return result


def main() -> None:
    print("=" * 90)
    print("Benchmark: check_static_eligibility with mock IR structures")
    print("=" * 90)

    results: list[dict] = []

    print("\n--- Eligible functions (varying call counts) ---")
    for n in CALL_COUNTS:
        r = bench_eligible(n)
        results.append(r)
        p = r["parameters"]
        print(
            f"  num_calls={n:4d}: mean={r['wall_time_ms']:.4f} ms, "
            f"eligible={p['is_eligible']}, tasks={p['num_tasks']}"
        )

    print("\n--- Ineligible functions (contain ForStmt) ---")
    for n in CALL_COUNTS:
        r = bench_ineligible(n)
        results.append(r)
        p = r["parameters"]
        print(
            f"  num_calls={n:4d}: mean={r['wall_time_ms']:.4f} ms, "
            f"eligible={p['is_eligible']}, reasons={p['num_reasons']}"
        )

    print("\n--- Multi-function Programs ---")
    program_configs = [
        (1, 10),
        (2, 10),
        (5, 10),
        (1, 50),
        (2, 50),
        (5, 50),
        (1, 100),
    ]
    for num_funcs, calls_per in program_configs:
        r = bench_program(num_funcs, calls_per)
        results.append(r)
        p = r["parameters"]
        print(
            f"  funcs={num_funcs}, calls/func={calls_per:3d}: "
            f"mean={r['wall_time_ms']:.4f} ms, "
            f"eligible={p['is_eligible']}, total_tasks={p['total_tasks']}"
        )

    path = save_results(results, "bench_eligibility.json")
    print(f"\nResults saved to: {path}")
    print(f"Total benchmarks: {len(results)}")


if __name__ == "__main__":
    main()
