# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Experimental static execution planning helpers for PyPTO Sonata."""

from .dependencies import (
    DEPENDENCY_POLICY_DATAFLOW_V0,
    DEPENDENCY_POLICY_SEQUENTIAL_V0,
    build_dataflow_dependencies,
    build_dependencies,
    build_sequential_dependencies,
    supports_dataflow_dependencies,
)
from .eligibility import check_static_eligibility
from .fallback import FallbackCode, code_for_reason
from .score import (
    DEFAULT_RUNTIME_TARGET,
    Dependency,
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
    eligibility_result_to_dict,
    score_fingerprint,
    score_to_dict,
    score_to_json,
)
from .storage import STORAGE_COVERAGE_REJECT_THRESHOLD, STORAGE_COVERAGE_WARN_THRESHOLD

__all__ = [
    "DEPENDENCY_POLICY_DATAFLOW_V0",
    "DEPENDENCY_POLICY_SEQUENTIAL_V0",
    "DEFAULT_RUNTIME_TARGET",
    "Dependency",
    "EligibilityResult",
    "ELIGIBILITY_RESULT_SCHEMA_VERSION",
    "FallbackCode",
    "FallbackReason",
    "RuntimeTarget",
    "SCORE_SCHEMA_VERSION",
    "STORAGE_COVERAGE_REJECT_THRESHOLD",
    "STORAGE_COVERAGE_WARN_THRESHOLD",
    "Score",
    "ShapeAssumption",
    "Task",
    "build_dataflow_dependencies",
    "build_dependencies",
    "build_sequential_dependencies",
    "check_static_eligibility",
    "code_for_reason",
    "eligibility_result_to_dict",
    "score_fingerprint",
    "score_to_dict",
    "score_to_json",
    "supports_dataflow_dependencies",
]
