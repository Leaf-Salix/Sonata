# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Stable fallback reason codes for Sonata eligibility and validation."""

from enum import Enum


class FallbackCode(str, Enum):
    """Stable reason codes for eligibility rejection and validation failure.

    These codes are safe to use as routing or filtering keys. They do not
    change when message wording changes.
    """

    UNSUPPORTED_ROOT_KIND = "unsupported_root_kind"
    CONTROL_FLOW_NOT_SUPPORTED = "control_flow_not_supported"
    UNSUPPORTED_RUNTIME_SCOPE = "unsupported_runtime_scope"
    TENSOR_READ_NOT_SUPPORTED = "tensor_read_not_supported"
    ENTRY_FUNCTION_NOT_ORCHESTRATION = "entry_function_not_orchestration"
    SCORE_VALIDATION_FAILED = "score_validation_failed"
    # Reserved for C-line (storage coverage)
    STORAGE_COVERAGE_BELOW_THRESHOLD = "storage_coverage_below_threshold"
    UNKNOWN_MEMORY_STORAGE_CRITICAL = "unknown_memory_storage_critical"
    # Reserved for D-line (dataflow fallback)
    DATAFLOW_DIRECTIONS_UNAVAILABLE = "dataflow_directions_unavailable"
    DATAFLOW_DIRECTIONS_INCOMPLETE = "dataflow_directions_incomplete"


_CONTROL_FLOW_KINDS = {"ForStmt", "IfStmt", "WhileStmt"}

# Mapping from eligibility rejection message patterns to FallbackCode.
# Patterns are matched as substrings (``pattern in message``).
# ORDER MATTERS: more specific patterns must come before shorter ones that
# could be substrings of them.
_ELIGIBILITY_CODE_MAP: list[tuple[str, FallbackCode]] = [
    # eligibility.py rejection messages
    ("entry function is not an orchestration function:", FallbackCode.ENTRY_FUNCTION_NOT_ORCHESTRATION),
    ("unsupported root for Sonata eligibility:", FallbackCode.UNSUPPORTED_ROOT_KIND),
    ("tensor.read calls are not supported by initial Sonata eligibility", FallbackCode.TENSOR_READ_NOT_SUPPORTED),
    ("is not supported by initial Sonata eligibility", FallbackCode.CONTROL_FLOW_NOT_SUPPORTED),
    # score.py Score.validate() messages
    ("score name must not be empty", FallbackCode.SCORE_VALIDATION_FAILED),
    ("task ids must be unique", FallbackCode.SCORE_VALIDATION_FAILED),
    ("task id must be non-negative:", FallbackCode.SCORE_VALIDATION_FAILED),
    ("func_id must be non-negative", FallbackCode.SCORE_VALIDATION_FAILED),
    ("has unsupported core_type:", FallbackCode.SCORE_VALIDATION_FAILED),
    ("arg_directions size", FallbackCode.SCORE_VALIDATION_FAILED),
    ("arg_storage_keys size", FallbackCode.SCORE_VALIDATION_FAILED),
    ("dependency producer is unknown:", FallbackCode.SCORE_VALIDATION_FAILED),
    ("dependency consumer is unknown:", FallbackCode.SCORE_VALIDATION_FAILED),
    ("dependency cannot be a self-edge:", FallbackCode.SCORE_VALIDATION_FAILED),
    ("dependency graph must be acyclic,", FallbackCode.SCORE_VALIDATION_FAILED),
    ("shape assumption symbol must not be empty", FallbackCode.SCORE_VALIDATION_FAILED),
    ("shape assumption symbol must be unique:", FallbackCode.SCORE_VALIDATION_FAILED),
    ("shape assumption", FallbackCode.SCORE_VALIDATION_FAILED),
]


def code_for_reason(message: str) -> FallbackCode | None:
    """Map an eligibility rejection message to a stable FallbackCode.

    Returns ``None`` if the message has no known enum mapping. The caller
    should fall back to ``_reason_code()`` in that case.
    """
    for pattern, code in _ELIGIBILITY_CODE_MAP:
        if pattern in message:
            if code == FallbackCode.CONTROL_FLOW_NOT_SUPPORTED:
                return _disambiguate_unsupported_kind(message)
            return code
    return None


def _disambiguate_unsupported_kind(message: str) -> FallbackCode:
    """Distinguish control-flow kinds from RuntimeScopeStmt."""
    kind = message.split(" is not supported")[0].strip()
    if kind in _CONTROL_FLOW_KINDS:
        return FallbackCode.CONTROL_FLOW_NOT_SUPPORTED
    if kind == "RuntimeScopeStmt":
        return FallbackCode.UNSUPPORTED_RUNTIME_SCOPE
    return FallbackCode.CONTROL_FLOW_NOT_SUPPORTED


__all__ = [
    "FallbackCode",
    "code_for_reason",
]
