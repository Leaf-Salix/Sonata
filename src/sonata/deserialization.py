# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Deserialization helpers: dict/JSON → Sonata data model objects.

Provides the reverse path for :mod:`sonata.serialization`, completing the
serialize/deserialize round-trip for Score, PlanHandle, and EligibilityResult.
"""

import json
from typing import Any

from .plan_handle import (
    FuncRegistry,
    FuncRegistryEntry,
    GuardStatus,
    PLAN_HANDLE_SCHEMA_VERSION,
    PlanHandle,
    RUNTIME_CONTRACT_VERSION,
    RuntimeArgBinding,
)
from .score import (
    Dependency,
    DependencyKind,
    EligibilityResult,
    FallbackReason,
    RuntimeTarget,
    Score,
    ShapeAssumption,
    Task,
)
from .serialization import (
    ELIGIBILITY_RESULT_SCHEMA_VERSION,
    SCORE_SCHEMA_VERSION,
)


class DeserializationError(Exception):
    """Raised when a serialized payload cannot be reconstructed."""


def score_from_dict(data: dict[str, Any]) -> Score:
    """Reconstruct a Score from a dictionary produced by ``score_to_dict``."""
    _require_dict(data, "score")
    _check_schema_version(data, SCORE_SCHEMA_VERSION, "score")
    name = _require_str(data, "name", "score")
    runtime_target = _runtime_target_from_dict(data.get("runtime_target", {}))
    tasks = _tasks_from_list(data.get("tasks", []))
    deps = _dependencies_from_list(data.get("dependencies", []))
    shapes = _shape_assumptions_from_list(data.get("shape_assumptions", []))
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise DeserializationError("score.metadata must be a dict")
    return Score(
        name=name,
        runtime_target=runtime_target,
        tasks=tasks,
        dependencies=deps,
        shape_assumptions=shapes,
        metadata=metadata,
    )


def score_from_json(text: str) -> Score:
    """Reconstruct a Score from a JSON string produced by ``score_to_json``."""
    return score_from_dict(_parse_json(text, "score"))


def plan_handle_from_dict(data: dict[str, Any]) -> PlanHandle:
    """Reconstruct a PlanHandle from a dictionary produced by ``plan_handle_to_dict``."""
    _require_dict(data, "plan_handle")
    _check_schema_version(data, PLAN_HANDLE_SCHEMA_VERSION, "plan_handle")
    fp = _require_str(data, "score_fingerprint", "plan_handle")
    rt = _runtime_target_from_dict(data.get("runtime_target", {}))
    source = _require_str(data, "source_adapter", "plan_handle")
    contract_ver = data.get("runtime_contract_version", RUNTIME_CONTRACT_VERSION)
    if not isinstance(contract_ver, int):
        raise DeserializationError("plan_handle.runtime_contract_version must be int")
    registry = _func_registry_from_list(data.get("func_registry", []))
    bindings = _arg_bindings_from_list(data.get("arg_bindings", []))
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise DeserializationError("plan_handle.metadata must be a dict")
    # v0.11: Per-region guard status
    region_gs_raw = data.get("region_guard_status", {})
    region_guard_status: dict[str, GuardStatus] = {}
    if isinstance(region_gs_raw, dict):
        for k, v in region_gs_raw.items():
            try:
                region_guard_status[k] = GuardStatus(v)
            except ValueError:
                raise DeserializationError(f"plan_handle.region_guard_status[{k!r}]: invalid GuardStatus {v!r}")

    # v0.17: Top-level guard status and critical guards
    guard_status_raw = data.get("guard_status", "all_satisfied")
    try:
        guard_status = GuardStatus(guard_status_raw)
    except ValueError:
        raise DeserializationError(f"plan_handle.guard_status: invalid GuardStatus {guard_status_raw!r}")

    critical_guards_raw = data.get("critical_guards", [])
    critical_guards: tuple[Any, ...] = ()
    if critical_guards_raw:
        if not isinstance(critical_guards_raw, list):
            raise DeserializationError("plan_handle.critical_guards must be a list")
        from .guard import GuardCondition
        parsed = []
        for g in critical_guards_raw:
            if not isinstance(g, dict):
                raise DeserializationError(f"plan_handle.critical_guards: expected dict, got {type(g).__name__}")
            parsed.append(GuardCondition.from_dict(g))
        critical_guards = tuple(parsed)

    return PlanHandle(
        score_fingerprint=fp,
        runtime_target=rt,
        source_adapter=source,
        runtime_contract_version=contract_ver,
        func_registry=registry,
        arg_bindings=bindings,
        metadata=metadata,
        guard_status=guard_status,
        critical_guards=critical_guards,
        region_guard_status=region_guard_status,
    )


def plan_handle_from_json(text: str) -> PlanHandle:
    """Reconstruct a PlanHandle from a JSON string."""
    return plan_handle_from_dict(_parse_json(text, "plan_handle"))


def eligibility_result_from_dict(data: dict[str, Any]) -> EligibilityResult:
    """Reconstruct an EligibilityResult from a dictionary."""
    _require_dict(data, "eligibility_result")
    _check_schema_version(data, ELIGIBILITY_RESULT_SCHEMA_VERSION, "eligibility_result")
    eligible = data.get("eligible")
    if not isinstance(eligible, bool):
        raise DeserializationError("eligibility_result.eligible must be bool")
    reasons = tuple(data.get("reasons", ()))
    details = tuple(
        _fallback_reason_from_dict(d)
        for d in data.get("reason_details", [])
    )
    score_data = data.get("score")
    score = score_from_dict(score_data) if score_data is not None else None
    return EligibilityResult(
        eligible=eligible,
        score=score,
        reasons=reasons,
        reason_details=details,
    )


def eligibility_result_from_json(text: str) -> EligibilityResult:
    """Reconstruct an EligibilityResult from a JSON string."""
    return eligibility_result_from_dict(_parse_json(text, "eligibility_result"))


def _runtime_target_from_dict(data: Any) -> RuntimeTarget:
    if not isinstance(data, dict):
        data = {}
    return RuntimeTarget(
        runtime=str(data.get("runtime", "tensormap_and_ringbuffer")),
        function_name=str(data.get("function_name", "aicpu_orchestration_entry")),
        aicpu_thread_num=data.get("aicpu_thread_num"),
        config_comment=tuple(data.get("config_comment", ())),
    )


def _tasks_from_list(items: Any) -> tuple[Task, ...]:
    if not isinstance(items, list):
        raise DeserializationError("tasks must be a list")
    result = []
    for i, item in enumerate(items):
        _require_dict(item, f"tasks[{i}]")
        result.append(Task(
            task_id=_require_int(item, "task_id", f"tasks[{i}]"),
            func_id=_require_int(item, "func_id", f"tasks[{i}]"),
            core_type=_require_str(item, "core_type", f"tasks[{i}]"),
            args=tuple(item.get("args", ())),
            arg_directions=tuple(item.get("arg_directions", ())),
            arg_storage_keys=tuple(item.get("arg_storage_keys", ())),
            name=item.get("name"),
            outputs=tuple(item.get("outputs", ())),
            storage_effects=_storage_effects_from_list(item.get("storage_effects")),
        ))
    return tuple(result)


def _storage_effects_from_list(items: Any) -> tuple[Any, ...] | None:
    """Deserialize storage_effects list. Returns None if items is None or empty.

    Raises DeserializationError on invalid items (fail-closed).
    """
    if items is None:
        return None
    if not isinstance(items, list):
        raise DeserializationError("storage_effects must be a list")
    if not items:
        return None
    from .score import StorageEffect
    result = []
    for i, e in enumerate(items):
        if not isinstance(e, dict):
            raise DeserializationError(f"storage_effects[{i}] must be a dict, got {type(e).__name__}")
        if "buffer_id" not in e:
            raise DeserializationError(f"storage_effects[{i}] missing 'buffer_id'")
        if "kind" not in e:
            raise DeserializationError(f"storage_effects[{i}] missing 'kind'")
        result.append(StorageEffect(buffer_id=str(e["buffer_id"]), kind=str(e["kind"])))
    return tuple(result)


def _dependencies_from_list(items: Any) -> tuple[Dependency, ...]:
    if not isinstance(items, list):
        raise DeserializationError("dependencies must be a list")
    result = []
    for i, item in enumerate(items):
        _require_dict(item, f"dependencies[{i}]")
        result.append(Dependency(
            producer=_require_int(item, "producer", f"dependencies[{i}]"),
            consumer=_require_int(item, "consumer", f"dependencies[{i}]"),
            kind=DependencyKind.from_str(item.get("kind", "data")),
        ))
    return tuple(result)


def _shape_assumptions_from_list(items: Any) -> tuple[ShapeAssumption, ...]:
    if not isinstance(items, list):
        raise DeserializationError("shape_assumptions must be a list")
    result = []
    for i, item in enumerate(items):
        _require_dict(item, f"shape_assumptions[{i}]")
        dims = item.get("dims", ())
        if not isinstance(dims, (list, tuple)):
            raise DeserializationError(f"shape_assumptions[{i}].dims must be a list")
        # Get severity with default to "hard" for backward compatibility
        severity_str = item.get("severity", "hard")
        from .guard import GuardSeverity
        severity = GuardSeverity(severity_str)
        result.append(ShapeAssumption(
            symbol=_require_str(item, "symbol", f"shape_assumptions[{i}]"),
            dims=tuple(dims),
            severity=severity,
        ))
    return tuple(result)


def _func_registry_from_list(items: Any) -> FuncRegistry:
    if not isinstance(items, list):
        raise DeserializationError("func_registry must be a list")
    entries = []
    for i, item in enumerate(items):
        _require_dict(item, f"func_registry[{i}]")
        entries.append(FuncRegistryEntry(
            name=_require_str(item, "name", f"func_registry[{i}]"),
            sonata_func_id=_require_int(item, "sonata_func_id", f"func_registry[{i}]"),
            runtime_func_id=item.get("runtime_func_id"),
        ))
    return FuncRegistry(entries=tuple(entries))


def _arg_bindings_from_list(items: Any) -> tuple[RuntimeArgBinding, ...]:
    if not isinstance(items, list):
        raise DeserializationError("arg_bindings must be a list")
    result = []
    for i, item in enumerate(items):
        _require_dict(item, f"arg_bindings[{i}]")
        result.append(RuntimeArgBinding(
            task_id=_require_int(item, "task_id", f"arg_bindings[{i}]"),
            arg_index=_require_int(item, "arg_index", f"arg_bindings[{i}]"),
            storage_key=item.get("storage_key"),
            direction=_require_str(item, "direction", f"arg_bindings[{i}]"),
            runtime_handle=item.get("runtime_handle"),
        ))
    return tuple(result)


def _fallback_reason_from_dict(data: Any) -> FallbackReason:
    _require_dict(data, "reason_detail")
    return FallbackReason(
        code=_require_str(data, "code", "reason_detail"),
        message=_require_str(data, "message", "reason_detail"),
        severity=str(data.get("severity", "error")),
    )


def _parse_json(text: str, label: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise DeserializationError(f"invalid JSON for {label}: {exc}") from exc
    if not isinstance(data, dict):
        raise DeserializationError(f"{label} JSON must decode to a dict")
    return data


def _require_dict(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise DeserializationError(f"{label} must be a dict, got {type(value).__name__}")


def _require_str(data: dict[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise DeserializationError(f"{label}.{key} must be a string, got {type(value).__name__}")
    return value


def _require_int(data: dict[str, Any], key: str, label: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DeserializationError(f"{label}.{key} must be an int, got {type(value).__name__}")
    return value


def _check_schema_version(data: dict[str, Any], expected: int, label: str) -> None:
    version = data.get("schema_version")
    if version is not None and version != expected:
        raise DeserializationError(
            f"{label} schema version mismatch: expected {expected}, got {version}"
        )


__all__ = [
    "DeserializationError",
    "eligibility_result_from_dict",
    "eligibility_result_from_json",
    "plan_handle_from_dict",
    "plan_handle_from_json",
    "score_from_dict",
    "score_from_json",
]
