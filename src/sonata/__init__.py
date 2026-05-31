# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Experimental static execution planning helpers for PyPTO Sonata."""

from .adapters import (
    AdapterCapability,
    AdapterDescriptor,
    AdapterRegistry,
    POST_SIMPLIFY,
    POST_SIMPLIFY_WITH_SCOPE,
    PRE_RUNTIME,
    default_registry,
)
from .alias import (
    ALIAS_ALIAS,
    ALIAS_DISJOINT,
    ALIAS_INPLACE,
    ALIAS_VIEW,
    AliasRelation,
    analyze_aliases,
)
from .cache import CACHE_SCHEMA_VERSION, CacheEntry, ScoreCache, cached_score
from .dependencies import (
    DEPENDENCY_POLICY_DATAFLOW_V0,
    DEPENDENCY_POLICY_SEQUENTIAL_V0,
    build_dataflow_dependencies,
    build_dependencies,
    build_mixed_dependencies,
    build_ordering_dependencies,
    build_sequential_dependencies,
    dataflow_dependency_fallback_code,
    supports_dataflow_dependencies,
)
from .deserialization import (
    DeserializationError,
    eligibility_result_from_dict,
    eligibility_result_from_json,
    plan_handle_from_dict,
    plan_handle_from_json,
    score_from_dict,
    score_from_json,
)
from .eligibility import check_static_eligibility
from .fallback import FallbackCode, code_for_reason
from .liveness import BufferLifetime, StorageConflict, compute_lifetimes, find_conflicts
from .memory_plan import BufferAllocation, MemoryPlan, plan_memory
from .regions import (
    REGION_DYNAMIC,
    REGION_STATIC,
    Region,
    RegionMap,
    check_region_eligibility,
    extract_regions,
)
from .plan_handle import (
    FuncRegistry,
    FuncRegistryEntry,
    PLAN_HANDLE_SCHEMA_VERSION,
    PlanHandle,
    RUNTIME_CONTRACT_VERSION,
    RuntimeArgBinding,
)
from .runtime_adapter import (
    HostBuildGraphEdge,
    HostBuildGraphPlan,
    HostBuildGraphRuntimeAdapter,
    HostBuildGraphTask,
    RuntimeAdapterResult,
)
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
    FINGERPRINT_VERSION,
    SCORE_SCHEMA_VERSION,
    eligibility_result_to_dict,
    plan_handle_to_dict,
    plan_handle_to_json,
    score_fingerprint,
    score_to_dict,
    score_to_json,
)
from .storage import STORAGE_COVERAGE_REJECT_THRESHOLD, STORAGE_COVERAGE_WARN_THRESHOLD

__all__ = [
    "AdapterCapability",
    "AdapterDescriptor",
    "AdapterRegistry",
    "ALIAS_ALIAS",
    "ALIAS_DISJOINT",
    "ALIAS_INPLACE",
    "ALIAS_VIEW",
    "AliasRelation",
    "BufferAllocation",
    "BufferLifetime",
    "CACHE_SCHEMA_VERSION",
    "CacheEntry",
    "DEPENDENCY_POLICY_DATAFLOW_V0",
    "DEPENDENCY_POLICY_SEQUENTIAL_V0",
    "DEFAULT_RUNTIME_TARGET",
    "Dependency",
    "DeserializationError",
    "EligibilityResult",
    "ELIGIBILITY_RESULT_SCHEMA_VERSION",
    "FINGERPRINT_VERSION",
    "FallbackCode",
    "FallbackReason",
    "FuncRegistry",
    "FuncRegistryEntry",
    "HostBuildGraphEdge",
    "HostBuildGraphPlan",
    "HostBuildGraphRuntimeAdapter",
    "HostBuildGraphTask",
    "MemoryPlan",
    "PLAN_HANDLE_SCHEMA_VERSION",
    "PlanHandle",
    "POST_SIMPLIFY",
    "POST_SIMPLIFY_WITH_SCOPE",
    "PRE_RUNTIME",
    "RUNTIME_CONTRACT_VERSION",
    "RuntimeAdapterResult",
    "RuntimeArgBinding",
    "RuntimeTarget",
    "SCORE_SCHEMA_VERSION",
    "STORAGE_COVERAGE_REJECT_THRESHOLD",
    "STORAGE_COVERAGE_WARN_THRESHOLD",
    "Score",
    "ScoreCache",
    "ShapeAssumption",
    "StorageConflict",
    "Task",
    "REGION_DYNAMIC",
    "REGION_STATIC",
    "Region",
    "RegionMap",
    "analyze_aliases",
    "build_dataflow_dependencies",
    "build_dependencies",
    "build_mixed_dependencies",
    "build_ordering_dependencies",
    "build_sequential_dependencies",
    "cached_score",
    "check_static_eligibility",
    "check_region_eligibility",
    "code_for_reason",
    "compute_lifetimes",
    "dataflow_dependency_fallback_code",
    "default_registry",
    "eligibility_result_from_dict",
    "eligibility_result_from_json",
    "eligibility_result_to_dict",
    "extract_regions",
    "find_conflicts",
    "plan_handle_from_dict",
    "plan_handle_from_json",
    "plan_handle_to_dict",
    "plan_handle_to_json",
    "plan_memory",
    "score_fingerprint",
    "score_from_dict",
    "score_from_json",
    "score_to_dict",
    "score_to_json",
    "supports_dataflow_dependencies",
]
