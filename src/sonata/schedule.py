# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Sonata Schedule Contract — runtime-neutral execution intent.

Defines ``SonataScheduleContract`` and its component types as pure Python
dataclasses. The contract describes **what** static regions, explicit
dependencies, dynamic gaps, boundary tensors, guards, and late-binding
placeholders exist — without coupling to any specific runtime backend.

A schedule is produced by ``build_schedule()`` from a ``Score`` and
``SonataAnalysisResult``, and consumed by backend adapters such as
``HBGScheduleBackend`` or the future ``sonata_tensormap_hybrid``.
"""

from __future__ import annotations

import json as _json
import logging as _logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .score import Score
from .serialization import score_fingerprint as _score_fingerprint

if TYPE_CHECKING:
    from .pipeline import SonataAnalysisResult

_log = _logging.getLogger("sonata.schedule")

SONATA_SCHEDULE_SCHEMA_VERSION = 1
RUNTIME_CONTRACT = "sonata_schedule_v1"


class FallbackPolicy(Enum):
    """Execution policy when the Sonata schedule cannot be fully applied.

    Mirror of the roadmap hard constraint: PyPTO original path must always work.
    """
    PARTIAL_FALLBACK = "partial_fallback"
    REPLAN_WITH_LIMIT = "replan_with_limit"
    FAIL = "fail"


@dataclass(frozen=True)
class ScheduleGuard:
    """Structured guard condition in a schedule contract.

    Replaces untyped dict to match TensorRT/TRT/JAX-style typed guard models.
    """
    guard_id: str = ""
    kind: str = "shape_range"
    severity: str = "hard"
    target: str = "*"
    symbolic_name: str | None = None
    dimension: int | None = None
    min_value: int | None = None
    max_value: int | None = None
    expression: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None


@dataclass(frozen=True)
class ArgBinding:
    """Late-bound argument identity in a schedule task.

    Sonata identity (stable logical name) is separate from codegen-assigned
    ``runtime_slot`` (nullable until v0.24 binding).
    """

    arg_identity: str
    runtime_slot: int | None = None


@dataclass(frozen=True)
class ScheduledTask:
    """One task in a static scheduling region."""

    task_id: int
    kernel_identity: str
    func_id: int | None
    core_type: str
    args: tuple[ArgBinding, ...] = ()
    outputs: tuple[str, ...] = ()
    name: str | None = None


@dataclass(frozen=True)
class ScheduleDep:
    """One explicit dependency edge in a static region."""

    producer: int
    consumer: int
    kind: str = "data"


@dataclass(frozen=True)
class ScheduledRegion:
    """One region in a Sonata schedule.

    ``kind="static"`` regions carry explicit task and dependency lists.
    ``kind="dynamic"`` regions carry ``mode="backend_dynamic"`` and delegate
    to the runtime's dynamic path (e.g. TensorMap).
    """

    region_id: str
    kind: str  # "static" | "dynamic"
    mode: str | None = None  # "backend_dynamic" for dynamic regions
    tasks: tuple[ScheduledTask, ...] = ()
    deps: tuple[ScheduleDep, ...] = ()


@dataclass(frozen=True)
class RegionBoundary:
    """Tensor handoff between two adjacent regions."""

    from_region: str
    to_region: str
    tensors: tuple[str, ...] = ()
    policy: str = "materialize"


@dataclass(frozen=True)
class SonataScheduleContract:
    """Top-level schedule artifact — runtime-neutral execution intent.

    Produced by ``build_schedule()``. Serialized as ``sonata_schedule.json``.
    """

    schema_version: int = SONATA_SCHEDULE_SCHEMA_VERSION
    runtime_contract: str = RUNTIME_CONTRACT
    fingerprint: str = ""
    regions: tuple[ScheduledRegion, ...] = ()
    boundaries: tuple[RegionBoundary, ...] = ()
    guards: tuple[ScheduleGuard, ...] = ()
    fallback_policy: FallbackPolicy | None = FallbackPolicy.PARTIAL_FALLBACK
    max_replans: int = 8
    supported_platforms: tuple[str, ...] = ("host_build_graph",)
    memory_plan_ref: str | None = None
    memory_plan_fingerprint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "runtime_contract": self.runtime_contract,
            "fingerprint": self.fingerprint,
            "regions": [
                {
                    "region_id": r.region_id,
                    "kind": r.kind,
                    "mode": r.mode,
                    "tasks": [
                        {
                            "task_id": t.task_id,
                            "kernel_identity": t.kernel_identity,
                            "func_id": t.func_id,
                            "core_type": t.core_type,
                            "args": [
                                {"arg_identity": a.arg_identity, "runtime_slot": a.runtime_slot}
                                for a in t.args
                            ],
                            "outputs": list(t.outputs),
                            "name": t.name,
                        }
                        for t in r.tasks
                    ] if r.tasks else [],
                    "deps": [
                        {"producer": d.producer, "consumer": d.consumer, "kind": d.kind}
                        for d in r.deps
                    ] if r.deps else [],
                }
                for r in self.regions
            ],
            "boundaries": [
                {
                    "from_region": b.from_region,
                    "to_region": b.to_region,
                    "tensors": list(b.tensors),
                    "policy": b.policy,
                }
                for b in self.boundaries
            ],
            "guards": [
                {
                    "guard_id": g.guard_id,
                    "kind": g.kind,
                    "severity": g.severity,
                    "target": g.target,
                    "symbolic_name": g.symbolic_name,
                    "dimension": g.dimension,
                    "min_value": g.min_value,
                    "max_value": g.max_value,
                    "expression": g.expression,
                    "failure_code": g.failure_code,
                    "failure_message": g.failure_message,
                }
                for g in self.guards
            ],
            "fallback_policy": self.fallback_policy.value if self.fallback_policy else None,
            "max_replans": self.max_replans,
            "supported_platforms": list(self.supported_platforms),
            "memory_plan_ref": self.memory_plan_ref,
            "memory_plan_fingerprint": self.memory_plan_fingerprint,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SonataScheduleContract":
        regions = tuple(
            ScheduledRegion(
                region_id=r["region_id"],
                kind=r["kind"],
                mode=r.get("mode"),
                tasks=tuple(
                    ScheduledTask(
                        task_id=t["task_id"],
                        kernel_identity=t["kernel_identity"],
                        func_id=t.get("func_id"),
                        core_type=t["core_type"],
                        args=tuple(
                            ArgBinding(
                                arg_identity=a["arg_identity"],
                                runtime_slot=a.get("runtime_slot"),
                            )
                            for a in t.get("args", [])
                        ),
                        outputs=tuple(t.get("outputs", [])),
                        name=t.get("name"),
                    )
                    for t in r.get("tasks", [])
                ),
                deps=tuple(
                    ScheduleDep(
                        producer=d["producer"],
                        consumer=d["consumer"],
                        kind=d.get("kind", "data"),
                    )
                    for d in r.get("deps", [])
                ),
            )
            for r in data.get("regions", [])
        )
        boundaries = tuple(
            RegionBoundary(
                from_region=b["from_region"],
                to_region=b["to_region"],
                tensors=tuple(b.get("tensors", [])),
                policy=b.get("policy", "materialize"),
            )
            for b in data.get("boundaries", [])
        )
        guards = tuple(
            ScheduleGuard(
                guard_id=g.get("guard_id", ""),
                kind=g.get("kind", "shape_range"),
                severity=g.get("severity", "hard"),
                target=g.get("target", "*"),
                symbolic_name=g.get("symbolic_name"),
                dimension=g.get("dimension"),
                min_value=g.get("min_value"),
                max_value=g.get("max_value"),
                expression=g.get("expression"),
                failure_code=g.get("failure_code"),
                failure_message=g.get("failure_message"),
            )
            for g in data.get("guards", [])
        )
        fp_raw = data.get("fallback_policy")
        if fp_raw is None:
            fallback_policy = FallbackPolicy.PARTIAL_FALLBACK
        elif isinstance(fp_raw, FallbackPolicy):
            fallback_policy = fp_raw
        else:
            try:
                fallback_policy = FallbackPolicy(fp_raw)
            except ValueError:
                _log.warning("Unknown fallback_policy %r, using PARTIAL_FALLBACK", fp_raw)
                fallback_policy = FallbackPolicy.PARTIAL_FALLBACK
        return cls(
            schema_version=data.get("schema_version", SONATA_SCHEDULE_SCHEMA_VERSION),
            runtime_contract=data.get("runtime_contract", RUNTIME_CONTRACT),
            fingerprint=data.get("fingerprint", ""),
            regions=regions,
            boundaries=boundaries,
            guards=guards,
            fallback_policy=fallback_policy,
            max_replans=data.get("max_replans", 8),
            supported_platforms=tuple(data.get("supported_platforms", ("host_build_graph",))),
            memory_plan_ref=data.get("memory_plan_ref"),
            memory_plan_fingerprint=data.get("memory_plan_fingerprint"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, indent: int = 2) -> str:
        return _json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "SonataScheduleContract":
        return cls.from_dict(_json.loads(text))

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def read_json(cls, path: str | Path) -> "SonataScheduleContract":
        return cls.from_json(Path(path).read_text())


def build_schedule(
    score: Score,
    analysis_result: SonataAnalysisResult,
) -> SonataScheduleContract:
    """Build a ``SonataScheduleContract`` from Sonata analysis outputs.

    Args:
        score: Computation identity with tasks and dependencies.
        analysis_result: Sonata analysis result with region statuses.

    Returns:
        A runtime-neutral schedule contract.
    """
    region_statuses: dict[str, str] = getattr(analysis_result, "region_statuses", {}) or {}
    region_tree = getattr(analysis_result, "region_tree", None)

    scheduled_regions: list[ScheduledRegion] = []
    boundaries: list[RegionBoundary] = []
    seen_region_ids: list[str] = []
    static_count = 0

    if not region_statuses:
        return SonataScheduleContract(
            fingerprint=_score_fingerprint(score),
            guards=_serialize_guards(score),
            fallback_policy=_fallback_from_analysis(analysis_result),
        )

    for region_id, status in region_statuses.items():
        if status == "static":
            static_count += 1
            region_node_ids = _get_region_nodes(region_id, region_tree)
            _log.info(
                "[build_schedule] region %s: kind=%s, task_ids=%s",
                region_id, status, sorted(region_node_ids) if region_node_ids else "(all tasks)",
            )
            region = _build_static_region(region_id, score, region_node_ids=region_node_ids)
        else:
            region = ScheduledRegion(
                region_id=region_id,
                kind="dynamic",
                mode="backend_dynamic",
            )
        scheduled_regions.append(region)
        seen_region_ids.append(region_id)

    if len(seen_region_ids) >= 2:
        for i in range(len(seen_region_ids) - 1):
            src = scheduled_regions[i]
            dst = scheduled_regions[i + 1]
            if src.kind != dst.kind:
                boundary_tensors = _collect_boundary_tensors(src, dst)
                if boundary_tensors:
                    boundaries.append(
                        RegionBoundary(
                            from_region=src.region_id,
                            to_region=dst.region_id,
                            tensors=boundary_tensors,
                        )
                    )

    return SonataScheduleContract(
        fingerprint=_score_fingerprint(score),
        regions=tuple(scheduled_regions),
        boundaries=tuple(boundaries),
        guards=_serialize_guards(score),
        fallback_policy=_fallback_from_analysis(analysis_result),
    )


def _build_static_region(
    region_id: str,
    score: Score,
    region_node_ids: set[int] | None = None,
) -> ScheduledRegion:
    task_filter = region_node_ids or {t.task_id for t in score.tasks}
    tasks = tuple(
        ScheduledTask(
            task_id=t.task_id,
            kernel_identity=t.name or f"task_{t.task_id}",
            func_id=t.func_id if t.func_id is not None else t.task_id,
            core_type=t.core_type,
            args=tuple(
                ArgBinding(
                    arg_identity=(
                        t.arg_storage_keys[i]
                        if hasattr(t, "arg_storage_keys") and i < len(t.arg_storage_keys) and t.arg_storage_keys[i]
                        else _fallback_arg_identity(t, i)
                    ),
                )
                for i in range(len(t.args))
            ),
            outputs=tuple(t.outputs) if hasattr(t, "outputs") and t.outputs else (),
            name=t.name,
        )
        for t in score.tasks
        if t.task_id in task_filter
    )
    deps = tuple(
        ScheduleDep(
            producer=d.producer,
            consumer=d.consumer,
            kind=_dep_kind_str(d),
        )
        for d in score.dependencies
        if d.producer in task_filter and d.consumer in task_filter
    )
    return ScheduledRegion(
        region_id=region_id,
        kind="static",
        tasks=tasks,
        deps=deps,
    )


def _dep_kind_str(dep: Any) -> str:
    kind = getattr(dep, "kind", None)
    if kind is not None:
        return kind.value if hasattr(kind, "value") else str(kind)
    return "data"


def _serialize_guards(score: Score) -> tuple[ScheduleGuard, ...]:
    guards: list[ScheduleGuard] = []
    for i, sa in enumerate(getattr(score, "shape_assumptions", []) or []):
        sev = getattr(sa, "severity", None)
        symbol = getattr(sa, "symbol", "")
        dims = tuple(getattr(sa, "dims", ())) if hasattr(sa, "dims") else ()
        guard = ScheduleGuard(
            guard_id=f"sa_{i}",
            kind="shape_range",
            severity=str(sev) if sev is not None else "soft",
            target="*",
            symbolic_name=str(symbol) if symbol else None,
            min_value=dims[0] if len(dims) >= 1 else None,
            max_value=dims[-1] if len(dims) >= 1 else None,
        )
        guards.append(guard)
    return tuple(guards)


def _fallback_from_analysis(analysis_result: SonataAnalysisResult) -> FallbackPolicy | None:
    reasons = getattr(analysis_result, "fallback_reasons", None)
    if reasons:
        return FallbackPolicy.PARTIAL_FALLBACK
    return None


def _get_region_nodes(region_id: str, region_tree: Any) -> set[int] | None:
    """Extract task node IDs for a region from region_tree.

    Returns None when region_tree is unavailable (fall back to full task list).
    """
    if region_tree is None:
        return None
    try:
        if hasattr(region_tree, "nodes"):
            nodes = region_tree.nodes
        elif hasattr(region_tree, "get"):
            nodes = region_tree.get(region_id)
        else:
            nodes = None
        if nodes and hasattr(nodes, "__iter__"):
            return {int(n) if not isinstance(n, int) else n for n in nodes}
    except (TypeError, ValueError, AttributeError):
        pass
    return None


def _collect_boundary_tensors(
    src: ScheduledRegion,
    dst: ScheduledRegion,
) -> tuple[str, ...]:
    if not src.tasks:
        return ()
    all_src_outputs: set[str] = set()
    for t in src.tasks:
        all_src_outputs.update(t.outputs)
    if not all_src_outputs:
        return ()
    if dst.tasks:
        consumed: set[str] = set()
        for t in dst.tasks:
            for a in t.args:
                consumed.add(a.arg_identity)
        return tuple(all_src_outputs & consumed)
    return tuple(all_src_outputs)


def _fallback_arg_identity(task: Any, index: int) -> str:
    _log.warning(
        "Task %d arg %d: arg_storage_keys empty, using index-based identity",
        getattr(task, "task_id", -1), index,
    )
    return f"{getattr(task, 'task_id', -1)}:arg{index}"


__all__ = [
    "ArgBinding",
    "FallbackPolicy",
    "RUNTIME_CONTRACT",
    "RegionBoundary",
    "SONATA_SCHEDULE_SCHEMA_VERSION",
    "ScheduleDep",
    "ScheduleGuard",
    "ScheduledRegion",
    "ScheduledTask",
    "SonataScheduleContract",
    "build_schedule",
]
