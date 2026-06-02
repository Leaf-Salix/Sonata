# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Stable JSON-like serialization helpers for Sonata scores."""

from dataclasses import asdict
import json
from hashlib import sha256
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .plan_handle import PlanHandle

from .score import EligibilityResult, Score

SCORE_SCHEMA_VERSION = 1
ELIGIBILITY_RESULT_SCHEMA_VERSION = 1
FINGERPRINT_VERSION = 1


def plan_handle_to_dict(plan_handle: "PlanHandle") -> dict[str, Any]:
    """Return a deterministic JSON-like dictionary for ``plan_handle``."""
    return {
        "schema_version": plan_handle.schema_version,
        "score_fingerprint": plan_handle.score_fingerprint,
        "runtime_target": {
            "runtime": plan_handle.runtime_target.runtime,
            "function_name": plan_handle.runtime_target.function_name,
            "aicpu_thread_num": plan_handle.runtime_target.aicpu_thread_num,
            "config_comment": list(plan_handle.runtime_target.config_comment),
        },
        "source_adapter": plan_handle.source_adapter,
        "runtime_contract_version": plan_handle.runtime_contract_version,
        "func_registry": [
            {
                "name": entry.name,
                "sonata_func_id": entry.sonata_func_id,
                "runtime_func_id": entry.runtime_func_id,
            }
            for entry in plan_handle.func_registry.entries
        ],
        "arg_bindings": [
            {
                "task_id": b.task_id,
                "arg_index": b.arg_index,
                "storage_key": b.storage_key,
                "direction": b.direction,
                "runtime_handle": _json_like(b.runtime_handle),
            }
            for b in plan_handle.arg_bindings
        ],
        "metadata": _json_like(plan_handle.metadata),
        # Phase 4: Guard condition integration
        "guard_status": plan_handle.guard_status.value,
        "critical_guards": [
            _json_like(guard.to_dict()) for guard in plan_handle.critical_guards
        ],
        # v0.11: Per-region guard status
        "region_guard_status": {
            k: v.value for k, v in plan_handle.region_guard_status.items()
        },
    }


def plan_handle_to_json(plan_handle: "PlanHandle", *, indent: int | None = 2) -> str:
    """Return a stable JSON string for ``plan_handle``."""
    return json.dumps(plan_handle_to_dict(plan_handle), indent=indent, sort_keys=True)


def score_to_dict(score: Score) -> dict[str, Any]:
    """Return a deterministic JSON-like dictionary for ``score``."""
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "name": score.name,
        "runtime_target": {
            "runtime": score.runtime_target.runtime,
            "function_name": score.runtime_target.function_name,
            "aicpu_thread_num": score.runtime_target.aicpu_thread_num,
            "config_comment": list(score.runtime_target.config_comment),
        },
        "tasks": [
            {
                "task_id": task.task_id,
                "func_id": task.func_id,
                "core_type": task.core_type,
                "name": task.name,
                "args": [_json_like(arg) for arg in task.args],
                "arg_directions": list(task.arg_directions),
                "arg_storage_keys": [_json_like(key) for key in task.arg_storage_keys],
                "outputs": list(task.outputs),
            }
            for task in score.tasks
        ],
        "dependencies": [
            {
                "producer": dependency.producer,
                "consumer": dependency.consumer,
                "kind": dependency.kind,
            }
            for dependency in score.dependencies
        ],
        "shape_assumptions": [
            {
                "symbol": shape.symbol,
                "dims": list(shape.dims),
                "severity": str(shape.severity),
            }
            for shape in score.shape_assumptions
        ],
        "metadata": _json_like(score.metadata),
    }


def score_to_json(score: Score, *, indent: int | None = 2) -> str:
    """Return a stable JSON string for ``score``."""
    return json.dumps(score_to_dict(score), indent=indent, sort_keys=True)


def eligibility_result_to_dict(result: EligibilityResult) -> dict[str, Any]:
    """Return a deterministic JSON-like dictionary for an eligibility result."""
    return {
        "schema_version": ELIGIBILITY_RESULT_SCHEMA_VERSION,
        "eligible": result.eligible,
        "reasons": list(result.reasons),
        "reason_details": [
            {k: v for k, v in _json_like(asdict(reason)).items() if v is not None}
            for reason in result.reason_details
        ],
        "score": score_to_dict(result.score) if result.score is not None else None,
    }


def score_fingerprint(score: Score, *, include_metadata: bool = False) -> str:
    """Return a stable SHA-256 fingerprint for the score computation identity.

    ``include_metadata`` includes only ``Score.metadata`` audit/debug data. It
    does not include artifact identity fields such as ``runtime_target``.
    """
    payload = json.dumps(
        _fingerprint_payload(score, include_metadata=include_metadata),
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _fingerprint_payload(score: Score, *, include_metadata: bool) -> dict[str, Any]:
    data = score_to_dict(score)
    identity = {
        "name": data["name"],
        "tasks": data["tasks"],
        "dependencies": data["dependencies"],
        "shape_assumptions": data["shape_assumptions"],
    }
    if include_metadata:
        identity["metadata"] = data["metadata"]
    return {
        "fingerprint_version": FINGERPRINT_VERSION,
        "identity": identity,
    }


def _json_like(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple | list):
        return [_json_like(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_like(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return repr(value)


__all__ = [
    "ELIGIBILITY_RESULT_SCHEMA_VERSION",
    "FINGERPRINT_VERSION",
    "SCORE_SCHEMA_VERSION",
    "eligibility_result_to_dict",
    "plan_handle_to_dict",
    "plan_handle_to_json",
    "score_fingerprint",
    "score_to_dict",
    "score_to_json",
]
