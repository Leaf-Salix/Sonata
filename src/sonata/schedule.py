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

from .directions import normalize_direction
from .score import Score
from .serialization import score_fingerprint as _score_fingerprint

if TYPE_CHECKING:
    from .pipeline import SonataAnalysisResult

_log = _logging.getLogger("sonata.schedule")

SONATA_SCHEDULE_SCHEMA_VERSION = 2
RUNTIME_CONTRACT = "sonata_schedule_v2"


class ArgDirection(Enum):
    """Direction of a tensor or scalar argument in a task.

    Maps 1:1 to TMARB ``TensorArgType`` and ``Arg::add_*()`` methods.
    Values are canonicalized to match ``normalize_direction()`` output.
    """
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"
    OUTPUT_EXISTING = "outputexisting"
    NO_DEP = "nodep"
    SCALAR = "scalar"


class ScopeMode(Enum):
    """Intent declaration for TMARB ``PTO2_SCOPE`` mode.

    ``AUTO``: TensorMap + explicit deps coexist (default).
    ``MANUAL``: explicit deps only.
    """
    AUTO = "auto"
    MANUAL = "manual"


@dataclass(frozen=True)
class MixedKernels:
    """AIC + AIV kernel IDs for a Group task.

    Maps to TMARB ``MixedKernels{aic_id, aiv_id, dual_aiv_id}``.
    """
    aic_func_id: int
    aiv_func_id: int
    dual_aiv_func_id: int | None = None


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

    ``direction`` (v0.25) tells the TMARB consumer how to dispatch:
    ``INPUT`` → ``add_input``, ``OUTPUT`` → ``add_output(ci)``, etc.
    """

    arg_identity: str
    runtime_slot: int | None = None
    direction: ArgDirection = ArgDirection.INPUT


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
    mixed_kernels: MixedKernels | None = None


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
    ``kind="dynamic"`` regions carry ``dynamic_mode="backend_dynamic"``
    and delegate to the runtime's dynamic path (e.g. TensorMap).

    ``scope_mode`` (v0.25) is Sonata's intent declaration for the TMARB
    ``PTO2_SCOPE`` mode. The backend MAY override it.
    """

    region_id: str
    kind: str  # "static" | "dynamic"
    dynamic_mode: str | None = None  # "backend_dynamic" for dynamic regions
    tasks: tuple[ScheduledTask, ...] = ()
    deps: tuple[ScheduleDep, ...] = ()
    scope_mode: ScopeMode = ScopeMode.AUTO


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
                    "dynamic_mode": r.dynamic_mode,
                    "scope_mode": r.scope_mode.value,
                    "tasks": [
                        {
                            "task_id": t.task_id,
                            "kernel_identity": t.kernel_identity,
                            "func_id": t.func_id,
                            "core_type": t.core_type,
                            "args": [
                                {
                                    "arg_identity": a.arg_identity,
                                    "runtime_slot": a.runtime_slot,
                                    "direction": a.direction.value if a.direction != ArgDirection.INPUT else None,
                                }
                                for a in t.args
                            ],
                            "outputs": list(t.outputs),
                            "name": t.name,
                            "mixed_kernels": (
                                {
                                    "aic_func_id": mk.aic_func_id,
                                    "aiv_func_id": mk.aiv_func_id,
                                    "dual_aiv_func_id": mk.dual_aiv_func_id,
                                } if t.mixed_kernels is not None else None
                            ),
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
        schema_ver = data.get("schema_version", 1)
        regions = tuple(
            cls._region_from_dict(r, schema_ver)
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
    @classmethod
    def _region_from_dict(cls, r: dict[str, Any], schema_ver: int) -> ScheduledRegion:
        if schema_ver < 2:
            raw_mode = r.get("mode")
        else:
            raw_mode = r.get("dynamic_mode")
            v1_mode = r.get("mode")
            if v1_mode is not None and raw_mode is None:
                _log.warning("Deprecated key 'mode' in region %r; use 'dynamic_mode'", r.get("region_id", "?"))
                raw_mode = v1_mode
        raw_scope = r.get("scope_mode")
        try:
            scope_mode = ScopeMode(raw_scope) if raw_scope else ScopeMode.AUTO
        except ValueError:
            scope_mode = ScopeMode.AUTO
        return ScheduledRegion(
            region_id=r["region_id"],
            kind=r["kind"],
            dynamic_mode=raw_mode,
            tasks=cls._tasks_from_dict(r.get("tasks", [])),
            deps=tuple(
                ScheduleDep(producer=d["producer"], consumer=d["consumer"], kind=d.get("kind", "data"))
                for d in r.get("deps", [])
            ),
            scope_mode=scope_mode,
        )

    @classmethod
    def _tasks_from_dict(cls, tasks: list[dict[str, Any]]) -> tuple[ScheduledTask, ...]:
        result: list[ScheduledTask] = []
        for t in tasks:
            mk_raw = t.get("mixed_kernels")
            mk = MixedKernels(**mk_raw) if mk_raw and isinstance(mk_raw, dict) else None
            result.append(ScheduledTask(
                task_id=t["task_id"],
                kernel_identity=t["kernel_identity"],
                func_id=t.get("func_id"),
                core_type=t["core_type"],
                args=cls._args_from_dict(t.get("args", [])),
                outputs=tuple(t.get("outputs", [])),
                name=t.get("name"),
                mixed_kernels=mk,
            ))
        return tuple(result)

    @classmethod
    def _args_from_dict(cls, args: list[dict[str, Any]]) -> tuple[ArgBinding, ...]:
        result: list[ArgBinding] = []
        for a in args:
            direction = ArgDirection.INPUT
            raw_dir = a.get("direction")
            if raw_dir is not None:
                try:
                    direction = ArgDirection(normalize_direction(raw_dir))
                except ValueError:
                    pass
            result.append(ArgBinding(arg_identity=a["arg_identity"], runtime_slot=a.get("runtime_slot"), direction=direction))
        return tuple(result)

    def to_binary(self) -> bytes:
        """Serialize to flat binary format for the sonata_tmarb interpreter.

        The binary layout matches ``flat_schedule.h`` struct layout:
        ``FlatSchedule + FlatRegion[] + FlatTask[] + FlatArg[] + FlatDep[]``
        """
        import struct

        # Direction → int16
        _DIR = {"input": 0, "output": 1, "inout": 2, "nodep": 3, "no_dep": 3, "scalar": 4}
        # Core type → int16
        _CORE = {"aic": 0, "aiv": 1, "mixed": 2}

        # Collect all tasks and deps
        flat_regions: list[bytes] = []
        flat_tasks: list[bytes] = []
        flat_args: list[bytes] = []
        flat_deps: list[bytes] = []
        task_cursor = 0
        dep_cursor = 0

        for region in self.regions:
            rkind = 0 if region.kind == "static" else 1
            rscope = 0 if region.scope_mode.value == "auto" else 1
            num_tasks = len(region.tasks)
            num_deps = len(region.deps)

            flat_regions.append(struct.pack("<iiiiii", rkind, rscope,
                                           task_cursor, num_tasks,
                                           dep_cursor, num_deps))

            for task in region.tasks:
                core = _CORE.get(task.core_type, 0)
                num_args = len(task.args)
                flat_tasks.append(struct.pack("<iihhii",
                    task.task_id,
                    task.func_id if task.func_id is not None else -1,
                    core, num_args,
                    dep_cursor, len(region.deps)))

                for arg in task.args:
                    d = _DIR.get(arg.direction.value if hasattr(arg.direction, "value") else arg.direction, 0)
                    flat_args.append(struct.pack("<ih", arg.runtime_slot or 0, d))

            for dep in region.deps:
                flat_deps.append(struct.pack("<ii", dep.producer, dep.consumer))

            task_cursor += num_tasks
            dep_cursor += num_deps

        # Build FlatSchedule header (72 bytes)
        magic = 0x534F4E41  # "SONA"
        fp_bytes = self.fingerprint.encode("utf-8")[:63]
        fp_bytes += b"\x00" * (64 - len(fp_bytes))
        header = struct.pack("<ii", magic, len(self.regions)) + fp_bytes

        return header + b"".join(flat_regions) + b"".join(flat_tasks) + b"".join(flat_args) + b"".join(flat_deps)

    @classmethod
    def from_binary(cls, data: bytes) -> "SonataScheduleContract":
        import struct
        _DIR = {0: "input", 1: "output", 2: "inout", 3: "nodep", 4: "scalar"}
        _CORE = {0: "aic", 1: "aiv", 2: "mixed"}
        if len(data) < 72: raise ValueError("too short")
        magic, nr = struct.unpack_from("<ii", data, 0)
        if magic != 0x534F4E41: raise ValueError(f"bad magic: {magic:#x}")
        fp = data[8:72].split(b"\x00", 1)[0].decode()
        rs = struct.calcsize("<iiiiii"); ts = struct.calcsize("<iihhii")
        ar = struct.calcsize("<ih"); ds = struct.calcsize("<ii")

        meta, off = [], 72
        for _ in range(nr):
            rk, _sc, _ts, tn, _ds, dn = struct.unpack_from("<iiiiii", data, off)
            meta.append({"kind": rk, "num_tasks": tn, "num_deps": dn})
            off += rs

        # Read task nargs from the task array (starts after all region headers)
        task_off = off
        for m in meta:
            m["tasks"] = []
            for ti in range(m["num_tasks"]):
                to = task_off + ti * ts
                tid, fid, co, na, _ds2, _dc2 = struct.unpack_from("<iihhii", data, to)
                m["tasks"].append({"tid": tid, "fid": fid, "core": co, "nargs": na})
            task_off += m["num_tasks"] * ts

        total_nargs = sum(t["nargs"] for m in meta for t in m["tasks"])
        t_off = 72 + nr * rs
        a_off = t_off + sum(m["num_tasks"] for m in meta) * ts
        d_off = a_off + total_nargs * ar

        regions, t_idx, dp, cum_args = [], 0, d_off, 0
        for i, m in enumerate(meta):
            tasks = []
            for t in m["tasks"]:
                args = []
                for ai in range(t["nargs"]):
                    ao = a_off + (cum_args + ai) * ar
                    sl, d = struct.unpack_from("<ih", data, ao)
                    args.append(ArgBinding(arg_identity=f"a{t_idx}_{ai}", runtime_slot=sl or None, direction=ArgDirection(_DIR.get(d, "input"))))
                tasks.append(ScheduledTask(task_id=t["tid"], kernel_identity=f"t{t['tid']}", func_id=t["fid"] if t["fid"] != -1 else None, core_type=_CORE.get(t["core"], "aic"), args=tuple(args)))
                cum_args += t["nargs"]
                t_idx += 1
            deps = []
            for _ in range(m["num_deps"]):
                p, c = struct.unpack_from("<ii", data, dp)
                deps.append(ScheduleDep(producer=p, consumer=c))
                dp += ds
            regions.append(ScheduledRegion(region_id=f"r{i}", kind="static" if m["kind"] == 0 else "dynamic", dynamic_mode=None if m["kind"] == 0 else "backend_dynamic", tasks=tuple(tasks), deps=tuple(deps)))
        return cls(fingerprint=fp, regions=tuple(regions))


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
                dynamic_mode="backend_dynamic",
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
            func_id=t.func_id,
            core_type=t.core_type,
            args=tuple(
                ArgBinding(
                    arg_identity=(
                        t.arg_storage_keys[i]
                        if hasattr(t, "arg_storage_keys") and i < len(t.arg_storage_keys) and t.arg_storage_keys[i]
                        else _fallback_arg_identity(t, i)
                    ),
                    direction=_resolve_direction(t, i),
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


def _resolve_direction(task: Any, index: int) -> ArgDirection:
    """Resolve ArgDirection from Score.Task.arg_directions."""
    raw = getattr(task, "arg_directions", None)
    if raw and index < len(raw) and raw[index]:
        canonical = normalize_direction(raw[index])
        try:
            return ArgDirection(canonical)
        except ValueError:
            pass
    return ArgDirection.INPUT


__all__ = [
    "ArgBinding",
    "ArgDirection",
    "FallbackPolicy",
    "MixedKernels",
    "RUNTIME_CONTRACT",
    "RegionBoundary",
    "SONATA_SCHEDULE_SCHEMA_VERSION",
    "ScheduleDep",
    "ScheduleGuard",
    "ScheduledRegion",
    "ScheduledTask",
    "ScopeMode",
    "SonataScheduleContract",
    "build_schedule",
]
