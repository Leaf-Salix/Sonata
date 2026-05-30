# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Conservative static-eligibility checks for Sonata planning.

This module intentionally uses structural introspection instead of importing IR
classes. The first pass should explain fallback reasons without adding hard
coupling to PyPTO's C++ bindings.
"""

from collections.abc import Iterable
from typing import Any

from .audit import build_score_metadata
from .dependencies import (
    DEPENDENCY_POLICY_DATAFLOW_V0,
    DEPENDENCY_POLICY_SEQUENTIAL_V0,
    build_dependencies,
    dataflow_dependency_fallback_code,
)
from .score import EligibilityResult, RuntimeTarget, Score, ShapeAssumption, Task, is_static_shape_dim
from .storage import (
    STORAGE_COVERAGE_WARN_THRESHOLD,
    arg_storage_keys,
    collect_call_output_vars,
    collect_storage_keys,
    propagate_call_output_storage,
)

_CONTROL_FLOW_KINDS = {"ForStmt", "IfStmt", "WhileStmt"}
_UNSUPPORTED_KINDS = {"RuntimeScopeStmt"}
_BUILTIN_OP_PREFIXES = ("tile.", "tensor.", "system.", "array.")


def check_static_eligibility(
    node: Any,
    *,
    runtime_target: RuntimeTarget | None = None,
    entry_name: str | None = None,
    dependency_policy: str = DEPENDENCY_POLICY_SEQUENTIAL_V0,
) -> EligibilityResult:
    """Return whether ``node`` is eligible for an initial Sonata static score."""
    reasons: list[str] = []
    root_kind = _kind(node)

    if root_kind not in {"Function", "Program"}:
        reasons.append(f"unsupported root for Sonata eligibility: {root_kind}")

    for child in _walk(node):
        child_kind = _kind(child)
        if child_kind in _CONTROL_FLOW_KINDS:
            reasons.append(f"{child_kind} is not supported by initial Sonata eligibility")
        elif child_kind in _UNSUPPORTED_KINDS:
            reasons.append(f"{child_kind} is not supported by initial Sonata eligibility")
        elif child_kind == "Call" and _call_name(child) == "tensor.read":
            reasons.append("tensor.read calls are not supported by initial Sonata eligibility")

    extraction_roots = _extraction_roots(node, entry_name)
    if entry_name is not None and not extraction_roots:
        reasons.append(f"entry function is not an orchestration function: {entry_name}")

    if reasons:
        return EligibilityResult.reject(*_dedupe(reasons))

    name = str(getattr(node, "name", "sonata_score"))
    tasks = _extract_tasks(extraction_roots, _function_core_types(node))
    resolved_policy, fallback_code = _resolve_dependency_policy(tasks, dependency_policy)
    score = Score(
        name=name,
        runtime_target=runtime_target
        or RuntimeTarget(runtime="host_build_graph", function_name=f"build_{name}_graph"),
        tasks=tasks,
        dependencies=build_dependencies(tasks, policy=resolved_policy),
        shape_assumptions=_extract_shape_assumptions(extraction_roots),
        metadata=build_score_metadata(
            extraction_roots,
            tasks,
            entry_name,
            resolved_policy,
            dependency_policy,
            fallback_code=fallback_code,
        ),
    )
    return _check_storage_coverage(score.validate())


def _check_storage_coverage(result: EligibilityResult) -> EligibilityResult:
    """Apply storage coverage checks after score validation passes."""
    if not result.eligible or result.score is None:
        return result

    coverage = result.score.metadata.get("memory_storage_key_coverage")
    if not coverage or coverage["total"] == 0:
        return result

    if coverage["unknown"] == 0:
        return result

    if coverage["known"] / coverage["total"] < STORAGE_COVERAGE_WARN_THRESHOLD:
        return EligibilityResult.accept_with_warnings(
            result.score,
            f"memory storage key coverage below threshold: "
            f"{coverage['known']}/{coverage['total']} < {STORAGE_COVERAGE_WARN_THRESHOLD}",
        )

    return result


def _walk(node: Any) -> Iterable[Any]:
    """Yield ``node`` and recursively walk common IR-like child fields."""
    seen: dict[int, Any] = {}
    stack = [node]
    child_fields = (
        "functions",
        "body",
        "then_body",
        "else_body",
        "branches",
        "stmts",
        "statements",
        "seq",
        "args",
        "value",
        "expr",
        "condition",
    )

    while stack:
        current = stack.pop()
        if current is None or isinstance(current, (str, bytes, int, float, bool)):
            continue
        ident = id(current)
        if ident in seen:
            continue
        seen[ident] = current
        yield current

        if isinstance(current, dict):
            stack.extend(current.values())
            continue
        if isinstance(current, (list, tuple)):
            stack.extend(reversed(current))
            continue

        for field in child_fields:
            if hasattr(current, field):
                stack.append(getattr(current, field))


def _kind(node: Any) -> str:
    return type(node).__name__


def _call_name(node: Any) -> str | None:
    op = getattr(node, "op", None)
    if isinstance(op, str):
        return op
    name = getattr(op, "name", None)
    if isinstance(name, str):
        return name
    op_name = getattr(node, "op_name", None)
    if isinstance(op_name, str):
        return op_name
    return None


def _is_builtin_call(call_name: str) -> bool:
    return call_name.startswith(_BUILTIN_OP_PREFIXES)


def _extract_tasks(nodes: tuple[Any, ...], core_types: dict[str, str]) -> tuple[Task, ...]:
    tasks: list[Task] = []
    func_ids: dict[str, int] = {}
    for node in nodes:
        storage_keys = collect_storage_keys(
            node,
            walk=_walk,
            kind=_kind,
            call_name=_call_name,
            is_builtin_call=_is_builtin_call,
            arg_name=_arg_name,
            arg_directions=_arg_directions,
        )
        call_outputs = collect_call_output_vars(node, walk=_walk, kind=_kind)
        for child in _walk(node):
            if _kind(child) != "Call":
                continue
            call_name = _call_name(child)
            if call_name is None or _is_builtin_call(call_name):
                continue
            if call_name not in func_ids:
                func_ids[call_name] = len(func_ids)
            tasks.append(
                Task(
                    task_id=len(tasks),
                    func_id=func_ids[call_name],
                    core_type=core_types.get(call_name, "mixed"),
                    args=tuple(_arg_name(arg) for arg in getattr(child, "args", ())),
                    arg_directions=_arg_directions(child),
                    arg_storage_keys=arg_storage_keys(child, storage_keys),
                    name=call_name,
                )
            )
            output_var = call_outputs.get(id(child))
            if output_var is not None:
                propagate_call_output_storage(output_var, child, storage_keys, arg_directions=_arg_directions)
    return tuple(tasks)


def _extract_shape_assumptions(nodes: tuple[Any, ...]) -> tuple[ShapeAssumption, ...]:
    assumptions: list[ShapeAssumption] = []
    multiple_roots = len(nodes) > 1
    for node in nodes:
        for param in getattr(node, "params", ()):
            symbol = _shape_symbol(node, param, multiple_roots)
            dims = _static_shape_dims(param)
            if symbol and dims is not None:
                assumptions.append(ShapeAssumption(symbol=symbol, dims=dims))
    return tuple(assumptions)


def _shape_symbol(root: Any, param: Any, multiple_roots: bool) -> str | None:
    symbol = _arg_name(param)
    if symbol == _kind(param):
        return None
    if not multiple_roots:
        return symbol
    root_name = getattr(root, "name", None)
    if isinstance(root_name, str) and root_name:
        return f"{root_name}.{symbol}"
    return symbol


def _static_shape_dims(param: Any) -> tuple[int, ...] | None:
    param_type = _param_type(param)
    if param_type is None:
        return None
    shape = _shape_values(param_type)
    if shape is None:
        return None
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Iterable):
        return None
    dims: list[int] = []
    for dim in shape:
        value = _const_int_value(dim)
        if value is None:
            return None
        dims.append(value)
    return tuple(dims)


def _param_type(param: Any) -> Any | None:
    for field in ("type", "type_", "tensor_type"):
        if hasattr(param, field):
            return getattr(param, field)
    return None


def _shape_values(param_type: Any) -> Any | None:
    for field in ("shape", "shape_", "dims", "dims_"):
        if hasattr(param_type, field):
            return getattr(param_type, field)
    return None


def _const_int_value(dim: Any) -> int | None:
    if is_static_shape_dim(dim):
        return dim
    value = getattr(dim, "value", None)
    if is_static_shape_dim(value):
        return value
    return None


def _extraction_roots(node: Any, entry_name: str | None) -> tuple[Any, ...]:
    functions = getattr(node, "functions", None)
    if not isinstance(functions, dict):
        if entry_name is not None and getattr(node, "name", None) != entry_name:
            return ()
        return (node,)

    roots: list[Any] = []
    for func in functions.values():
        func_type = getattr(func, "func_type", None)
        if getattr(func_type, "name", None) == "Orchestration" and (
            entry_name is None or getattr(func, "name", None) == entry_name
        ):
            roots.append(func)
    return tuple(roots)


def _resolve_dependency_policy(
    tasks: tuple[Task, ...], requested_policy: str
) -> tuple[str, "FallbackCode | None"]:
    from .fallback import FallbackCode

    if requested_policy == DEPENDENCY_POLICY_DATAFLOW_V0:
        code = dataflow_dependency_fallback_code(tasks)
        if code is not None:
            return DEPENDENCY_POLICY_SEQUENTIAL_V0, code
    return requested_policy, None


def _function_core_types(node: Any) -> dict[str, str]:
    functions = getattr(node, "functions", None)
    if not isinstance(functions, dict):
        return {}

    core_types: dict[str, str] = {}
    for func in functions.values():
        name = getattr(func, "name", None)
        if isinstance(name, str):
            core_types[name] = _core_type_from_function(func)
    return core_types


def _core_type_from_function(func: Any) -> str:
    func_type = getattr(func, "func_type", None)
    func_type_name = getattr(func_type, "name", None)
    if func_type_name == "AIC":
        return "aic"
    if func_type_name == "AIV":
        return "aiv"
    return "mixed"


def _arg_name(node: Any) -> str:
    if isinstance(node, str):
        return node
    name_hint = getattr(node, "name_hint", None)
    if isinstance(name_hint, str):
        return name_hint
    name = getattr(node, "name", None)
    if isinstance(name, str):
        return name
    return _kind(node)


def _arg_directions(call: Any) -> tuple[str, ...]:
    directions = getattr(call, "arg_directions", None)
    if directions is None:
        attrs = getattr(call, "attrs", None)
        if isinstance(attrs, dict):
            directions = attrs.get("arg_directions")
    if not directions:
        return ()
    return tuple(_direction_name(direction) for direction in directions)


def _direction_name(direction: Any) -> str:
    name = getattr(direction, "name", None)
    if isinstance(name, str):
        return name
    value = getattr(direction, "value", None)
    if isinstance(value, str):
        return value
    return str(direction)


def _dedupe(reasons: list[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            result.append(reason)
    return tuple(result)
