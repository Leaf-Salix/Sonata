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
    UNSUPPORTED_PYPTO_ADAPTER_SCOPE = "unsupported_pypto_adapter_scope"
    SCORE_VALIDATION_FAILED = "score_validation_failed"
    # Reserved for C-line (storage coverage)
    STORAGE_COVERAGE_BELOW_THRESHOLD = "storage_coverage_below_threshold"
    UNKNOWN_MEMORY_STORAGE_CRITICAL = "unknown_memory_storage_critical"
    # Reserved for D-line (dataflow fallback)
    DATAFLOW_DIRECTIONS_UNAVAILABLE = "dataflow_directions_unavailable"
    DATAFLOW_DIRECTIONS_INCOMPLETE = "dataflow_directions_incomplete"
    # Reserved for v0.2 runtime adapter
    RUNTIME_ADAPTER_FINGERPRINT_MISMATCH = "runtime_adapter_fingerprint_mismatch"
    RUNTIME_ADAPTER_CONTRACT_VERSION_MISMATCH = "runtime_adapter_contract_version_mismatch"
    RUNTIME_ADAPTER_FUNC_NOT_REGISTERED = "runtime_adapter_func_not_registered"
    RUNTIME_ADAPTER_FUNC_UNREFERENCED = "runtime_adapter_func_unreferenced"
    RUNTIME_ADAPTER_BINDING_INCOMPLETE = "runtime_adapter_binding_incomplete"
    RUNTIME_ADAPTER_INVALID_EDGE = "runtime_adapter_invalid_edge"


def code_for_reason(message: str) -> FallbackCode | None:
    """Return a legacy best-effort code mapping for ``message``.

    Stable fallback codes must be emitted at the reason producer by passing an
    explicit ``FallbackReason``. Raw strings intentionally have no stable enum
    mapping because message wording is not an API contract.
    """
    return None


__all__ = [
    "FallbackCode",
    "code_for_reason",
]
