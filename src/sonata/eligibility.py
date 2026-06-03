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
from .fallback import FallbackCode
from .pypto_adapter import PostSimplifyPyPTOInputAdapter, PyPTOAdapterContractError
from .score import (
    EligibilityResult,
    FallbackReason,
    RuntimeTarget,
    Score,
    ShapeAssumption,
    Task,
    is_static_shape_dim,
)
from .storage import (
    STORAGE_COVERAGE_WARN_THRESHOLD,
    arg_storage_keys,
    collect_call_output_vars,
    collect_storage_keys,
    propagate_call_output_storage,
)

_CONTROL_FLOW_KINDS = {"ForStmt", "IfStmt", "WhileStmt"}
_UNROLL_THRESHOLD = 16  # ForStmt with trip_count <= this is "unrollable"


def _is_unrollable_for_stmt(node: Any) -> bool:
    """Check if a ForStmt has constant trip count <= _UNROLL_THRESHOLD.

    A ForStmt is unrollable when:
    - It has numeric start, stop attributes (PyPTO IR range)
    - trip_count = (stop - start) is a positive integer <= threshold
    - step is 1 or absent (simple range)

    Returns False for any ForStmt that doesn't meet these criteria.
    """
    if type(node).__name__ != "ForStmt":
        return False

    start = getattr(node, "start", None)
    stop = getattr(node, "stop", None)
    step = getattr(node, "step", None)

    # start/stop must be numeric constants
    if not isinstance(start, (int, float)) or not isinstance(stop, (int, float)):
        return False
    if isinstance(start, bool) or isinstance(stop, bool):
        return False

    # step must be 1, None, or absent
    if step is not None and step != 1:
        return False

    trip_count = int(stop) - int(start)
    return 0 < trip_count <= _UNROLL_THRESHOLD


def check_static_eligibility(
    node: Any,
    *,
    runtime_target: RuntimeTarget | None = None,
    entry_name: str | None = None,
    dependency_policy: str = DEPENDENCY_POLICY_SEQUENTIAL_V0,
    require_certified: bool = False,
) -> EligibilityResult:
    """Return whether ``node`` is eligible for an initial Sonata static score."""
    reasons: list[FallbackReason] = []
    adapter = PostSimplifyPyPTOInputAdapter(node, entry_name=entry_name)
    root_kind = _kind(node)

    if root_kind not in {"Function", "Program"}:
        reasons.append(
            _fallback_reason(
                FallbackCode.UNSUPPORTED_ROOT_KIND,
                f"unsupported root for Sonata eligibility: {root_kind}",
            )
        )

    extraction_roots = adapter.extraction_roots()
    for root in extraction_roots:
        for child in adapter.walk(root):
            child_kind = _kind(child)
            if child_kind in _CONTROL_FLOW_KINDS:
                # v0.18 Phase 2 B1: Small constant ForStmt is unrollable
                if child_kind == "ForStmt" and _is_unrollable_for_stmt(child):
                    continue  # don't reject — treat as static
                reasons.append(
                    _fallback_reason(
                        FallbackCode.CONTROL_FLOW_NOT_SUPPORTED,
                        f"{child_kind} is not supported by initial Sonata eligibility",
                    )
                )
            elif child_kind == "RuntimeScopeStmt":
                reasons.append(
                    _fallback_reason(
                        FallbackCode.UNSUPPORTED_RUNTIME_SCOPE,
                        f"{child_kind} is not supported by initial Sonata eligibility",
                    )
                )
            elif child_kind in adapter.unsupported_kinds:
                reasons.append(
                    _fallback_reason(
                        FallbackCode.UNSUPPORTED_PYPTO_ADAPTER_SCOPE,
                        f"{child_kind} is out of scope for Sonata v0.1 PyPTO adapter",
                    )
                )
            elif child_kind == "Call" and _call_name(child) == "tensor.read":
                reasons.append(
                    _fallback_reason(
                        FallbackCode.TENSOR_READ_NOT_SUPPORTED,
                        "tensor.read calls are not supported by initial Sonata eligibility",
                    )
                )

    if entry_name is not None and not extraction_roots:
        reasons.append(
            _fallback_reason(
                FallbackCode.ENTRY_FUNCTION_NOT_ORCHESTRATION,
                f"entry function is not an orchestration function: {entry_name}",
            )
        )
    elif root_kind == "Program" and not extraction_roots:
        reasons.append(
            _fallback_reason(
                FallbackCode.ENTRY_FUNCTION_NOT_ORCHESTRATION,
                "program has no Orchestration functions for Sonata eligibility",
            )
        )
    root_error = adapter.root_out_of_scope_error()
    if root_error is not None:
        reasons.append(
            _fallback_reason(
                FallbackCode.UNSUPPORTED_PYPTO_ADAPTER_SCOPE,
                root_error,
            )
        )
    unsupported_call = adapter.has_unsupported_function_call()
    if unsupported_call is not None:
        reasons.append(
            _fallback_reason(
                FallbackCode.UNSUPPORTED_PYPTO_ADAPTER_SCOPE,
                f"Group/Spmd callee is out of scope for Sonata v0.1: {unsupported_call}",
            )
        )

    if reasons:
        return EligibilityResult.reject(*_dedupe(reasons))

    try:
        facts = adapter.normalize(require_certified=require_certified)
    except PyPTOAdapterContractError as exc:
        return EligibilityResult.reject(
            _fallback_reason(
                FallbackCode.UNSUPPORTED_PYPTO_ADAPTER_SCOPE,
                str(exc),
            )
        )

    name = str(getattr(node, "name", "sonata_score"))
    tasks = _tasks_from_facts(facts.functions)
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
            _fallback_reason(
                FallbackCode.STORAGE_COVERAGE_BELOW_THRESHOLD,
                f"memory storage key coverage below threshold: "
                f"{coverage['known']}/{coverage['total']} < {STORAGE_COVERAGE_WARN_THRESHOLD}",
                severity="warning",
            ),
        )

    return result


_walk_cache: dict[int, tuple[Any, ...]] = {}


def _walk(node: Any) -> Iterable[Any]:
    """Yield ``node`` and recursively walk common IR-like child fields.

    v0.17 Phase 2 C2: Results are cached by node identity so that multiple
    collectors (collect_storage_keys, collect_call_output_vars) sharing
    the same root node avoid redundant tree traversals.
    """
    node_id = id(node)
    cached = _walk_cache.get(node_id)
    if cached is not None:
        yield from cached
        return
    result = tuple(PostSimplifyPyPTOInputAdapter.walk(node))
    _walk_cache[node_id] = result
    yield from result


def _kind(node: Any) -> str:
    return type(node).__name__


def _fallback_reason(
    code: FallbackCode,
    message: str,
    *,
    severity: str = "error",
) -> FallbackReason:
    return FallbackReason(code=code.value, message=message, severity=severity)


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
    return PostSimplifyPyPTOInputAdapter.is_builtin_call(call_name)


def _tasks_from_facts(functions: tuple[Any, ...]) -> tuple[Task, ...]:
    tasks: list[Task] = []
    func_ids: dict[str, int] = {}
    multiple_roots = len(functions) > 1
    for root_index, function in enumerate(functions):
        storage_keys = collect_storage_keys(
            function.node,
            walk=_walk,
            kind=_kind,
            call_name=_call_name,
            is_builtin_call=_is_builtin_call,
            arg_name=_arg_name,
            arg_directions=_arg_directions,
        )
        if multiple_roots:
            storage_keys = _namespace_storage_keys(storage_keys, function.name, root_index)
        call_outputs = collect_call_output_vars(function.node, walk=_walk, kind=_kind)
        for call in function.calls:
            if call.callee_name not in func_ids:
                func_ids[call.callee_name] = len(func_ids)
            task = Task(
                task_id=len(tasks),
                func_id=func_ids[call.callee_name],
                core_type=call.core_type,
                args=call.arg_names,
                arg_directions=call.arg_directions,
                arg_storage_keys=arg_storage_keys(call.node, storage_keys),
                name=call.callee_name,
            )
            tasks.append(task)
            output_var = call_outputs.get(id(call.node))
            if output_var is not None:
                propagate_call_output_storage(output_var, call.node, storage_keys, arg_directions=_arg_directions)
    return tuple(tasks)


def _namespace_storage_keys(storage_keys: dict[int, str], root_name: str | None, root_index: int) -> dict[int, str]:
    namespace = root_name or f"root{root_index}"
    return {identity: _namespace_storage_key(key, namespace) for identity, key in storage_keys.items()}


def _namespace_storage_key(key: str, namespace: str) -> str:
    prefix, sep, rest = key.partition(":")
    if not sep:
        return f"{namespace}.{key}"
    return f"{prefix}:{namespace}.{rest}"


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


def _resolve_dependency_policy(
    tasks: tuple[Task, ...], requested_policy: str
) -> tuple[str, "FallbackCode | None"]:
    if requested_policy == DEPENDENCY_POLICY_DATAFLOW_V0:
        code = dataflow_dependency_fallback_code(tasks)
        if code is not None:
            return DEPENDENCY_POLICY_SEQUENTIAL_V0, code
    return requested_policy, None


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


def _dedupe(reasons: list[FallbackReason]) -> tuple[FallbackReason, ...]:
    result: list[FallbackReason] = []
    seen: set[tuple[str, str, str]] = set()
    for reason in reasons:
        key = (reason.code, reason.message, reason.severity)
        if key not in seen:
            seen.add(key)
            result.append(reason)
    return tuple(result)
