# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Region extraction and region-level eligibility for Sonata v0.4.

Splits an IR graph into static and dynamic regions. Static regions can
be independently planned by Sonata; dynamic regions fall back to the
original PyPTO runtime path. This replaces the v0.1 whole-graph
eligibility check with per-region granularity.
"""

from dataclasses import dataclass, field
from typing import Any

from .fallback import FallbackCode
from .score import EligibilityResult, FallbackReason, Score


REGION_STATIC = "static"
REGION_DYNAMIC = "dynamic"

_CONTROL_FLOW_KINDS = {"ForStmt", "IfStmt", "WhileStmt"}


@dataclass(frozen=True)
class Region:
    """One region of an IR graph, classified as static or dynamic."""

    region_id: int
    kind: str
    nodes: tuple[Any, ...] = ()
    control_flow_kind: str | None = None
    fallback_reason: FallbackReason | None = None

    @property
    def is_static(self) -> bool:
        return self.kind == REGION_STATIC

    @property
    def is_dynamic(self) -> bool:
        return self.kind == REGION_DYNAMIC

    @property
    def node_count(self) -> int:
        return len(self.nodes)


@dataclass(frozen=True)
class RegionMap:
    """Collection of regions extracted from an IR graph."""

    regions: tuple[Region, ...] = ()

    def static_regions(self) -> tuple[Region, ...]:
        return tuple(r for r in self.regions if r.is_static)

    def dynamic_regions(self) -> tuple[Region, ...]:
        return tuple(r for r in self.regions if r.is_dynamic)

    def static_ratio(self) -> float:
        if not self.regions:
            return 0.0
        return len(self.static_regions()) / len(self.regions)

    def all_static(self) -> bool:
        return all(r.is_static for r in self.regions)

    def all_dynamic(self) -> bool:
        return all(r.is_dynamic for r in self.regions)


def extract_regions(node: Any) -> RegionMap:
    """Extract static and dynamic regions from an IR node.

    Walks the node's body and splits statements into contiguous runs
    of static statements (no control flow) separated by dynamic
    statements (control flow nodes).
    """
    body = _get_body(node)
    if body is None:
        return RegionMap(regions=(_classify_node(node, 0),))

    regions: list[Region] = []
    static_buffer: list[Any] = []
    region_id = 0

    for stmt in body:
        kind = type(stmt).__name__
        if kind in _CONTROL_FLOW_KINDS:
            if static_buffer:
                regions.append(Region(
                    region_id=region_id,
                    kind=REGION_STATIC,
                    nodes=tuple(static_buffer),
                ))
                region_id += 1
                static_buffer = []

            regions.append(Region(
                region_id=region_id,
                kind=REGION_DYNAMIC,
                nodes=(stmt,),
                control_flow_kind=kind,
                fallback_reason=FallbackReason(
                    code=FallbackCode.CONTROL_FLOW_NOT_SUPPORTED.value,
                    message=f"{kind} region falls back to dynamic execution",
                    severity="warning",
                ),
            ))
            region_id += 1
        elif kind == "RuntimeScopeStmt":
            if static_buffer:
                regions.append(Region(
                    region_id=region_id,
                    kind=REGION_STATIC,
                    nodes=tuple(static_buffer),
                ))
                region_id += 1
                static_buffer = []
            regions.append(Region(
                region_id=region_id,
                kind=REGION_DYNAMIC,
                nodes=(stmt,),
                control_flow_kind=kind,
                fallback_reason=FallbackReason(
                    code=FallbackCode.UNSUPPORTED_RUNTIME_SCOPE.value,
                    message=f"{kind} region falls back to dynamic execution",
                    severity="warning",
                ),
            ))
            region_id += 1
        else:
            static_buffer.append(stmt)

    if static_buffer:
        regions.append(Region(
            region_id=region_id,
            kind=REGION_STATIC,
            nodes=tuple(static_buffer),
        ))

    if not regions:
        regions.append(Region(
            region_id=0,
            kind=REGION_STATIC,
            nodes=(),
        ))

    return RegionMap(regions=tuple(regions))


def check_region_eligibility(
    node: Any,
    *,
    entry_name: str | None = None,
) -> EligibilityResult:
    """Check eligibility at region granularity.

    Static regions are eligible for Sonata planning; dynamic regions
    fall back with structured warnings. A graph with at least one
    static region is partially eligible.
    """
    from .eligibility import check_static_eligibility

    region_map = extract_regions(node)

    if region_map.all_dynamic():
        reasons = []
        for r in region_map.dynamic_regions():
            if r.fallback_reason is not None:
                reasons.append(r.fallback_reason)
        if not reasons:
            reasons.append(FallbackReason(
                code=FallbackCode.CONTROL_FLOW_NOT_SUPPORTED.value,
                message="all regions are dynamic, no static region available",
            ))
        return EligibilityResult.reject(*reasons)

    return check_static_eligibility(node, entry_name=entry_name)


def _classify_node(node: Any, region_id: int) -> Region:
    body = _get_body(node)
    if body is None:
        kind = type(node).__name__
        if kind in _CONTROL_FLOW_KINDS:
            return Region(
                region_id=region_id,
                kind=REGION_DYNAMIC,
                nodes=(node,),
                control_flow_kind=kind,
                fallback_reason=FallbackReason(
                    code=FallbackCode.CONTROL_FLOW_NOT_SUPPORTED.value,
                    message=f"{kind} is a dynamic region",
                ),
            )
        return Region(
            region_id=region_id,
            kind=REGION_STATIC,
            nodes=(node,),
        )

    return extract_regions(node).regions[0] if extract_regions(node).regions else Region(
        region_id=region_id,
        kind=REGION_STATIC,
        nodes=(),
    )


def _get_body(node: Any) -> list[Any] | tuple[Any, ...] | None:
    for field_name in ("body", "stmts", "statements", "seq"):
        body = getattr(node, field_name, None)
        if isinstance(body, (list, tuple)) and body:
            return body
    return None


__all__ = [
    "REGION_DYNAMIC",
    "REGION_STATIC",
    "Region",
    "RegionMap",
    "check_region_eligibility",
    "extract_regions",
]
