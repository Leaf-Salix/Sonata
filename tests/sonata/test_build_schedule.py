"""Integration tests for build_schedule() — Score + SonataAnalysisResult → contract."""

import pytest
from dataclasses import dataclass, field
from typing import Any

from sonata.score import Dependency, RuntimeTarget, Score, Task
from sonata.schedule import (
    SonataScheduleContract,
    ScheduledRegion,
    ScheduledTask,
    build_schedule,
    SONATA_SCHEDULE_SCHEMA_VERSION,
)


@dataclass
class FakeSonataAnalysisResult:
    """Minimal fake of SonataAnalysisResult for testing build_schedule()."""
    region_statuses: dict[str, str] = field(default_factory=dict)
    region_tree: Any = None
    fallback_reasons: Any = None
    eligibility_result: Any = None


def _make_score(
    name="test_graph",
    tasks=None,
    deps=None,
    shape_assumptions=None,
) -> Score:
    return Score(
        name=name,
        runtime_target=RuntimeTarget(runtime="host_build_graph"),
        tasks=tuple(tasks or []),
        dependencies=tuple(deps or []),
        shape_assumptions=tuple(shape_assumptions or []),
    )


def _task(task_id, func_id=0, core_type="aic", args=None, storage_keys=None, name=None, outputs=None):
    return Task(
        task_id=task_id,
        func_id=func_id,
        core_type=core_type,
        args=tuple(args or []),
        arg_storage_keys=tuple(storage_keys or []),
        name=name,
        outputs=tuple(outputs or []),
    )


class TestBuildSchedule:
    def test_empty_score_no_regions(self):
        score = _make_score(name="empty")
        result = FakeSonataAnalysisResult()
        c = build_schedule(score, result)
        assert isinstance(c, SonataScheduleContract)
        assert c.regions == ()
        assert c.boundaries == ()
        assert c.fingerprint != ""

    def test_single_static_region_with_tasks(self):
        t1 = _task(0, func_id=10, core_type="aic", args=[1.0, 2.0],
            storage_keys=["x", "tmp"], outputs=["tmp"], name="load")
        t2 = _task(1, func_id=11, core_type="aiv", args=[3.0, 4.0],
            storage_keys=["tmp", "y"], outputs=["y"], name="store")
        score = _make_score(
            name="single_region",
            tasks=[t1, t2],
            deps=[Dependency(producer=0, consumer=1)],
        )
        result = FakeSonataAnalysisResult(region_statuses={"r0": "static"})
        c = build_schedule(score, result)

        assert len(c.regions) == 1
        r = c.regions[0]
        assert r.region_id == "r0"
        assert r.kind == "static"
        assert len(r.tasks) == 2
        assert len(r.deps) == 1
        assert r.tasks[0].kernel_identity == "load"
        assert r.tasks[0].func_id == 10
        assert r.tasks[0].args[0].arg_identity == "x"
        assert r.tasks[0].outputs == ("tmp",)
        assert r.tasks[1].kernel_identity == "store"

    def test_outputs_from_outputs_field(self):
        t1 = _task(0, func_id=1, core_type="aic", storage_keys=["x", "y", "z"], outputs=["z"])
        score = _make_score(name="out_test", tasks=[t1])
        result = FakeSonataAnalysisResult(region_statuses={"r0": "static"})
        c = build_schedule(score, result)

        assert len(c.regions[0].tasks) == 1
        scheduled = c.regions[0].tasks[0]
        assert scheduled.outputs == ("z",), f"expected ('z',), got {scheduled.outputs}"
        assert "z" in scheduled.outputs
        assert "x" not in scheduled.outputs
        assert "y" not in scheduled.outputs

    def test_mixed_static_and_dynamic_regions(self):
        t1 = _task(0, func_id=1, core_type="aic", storage_keys=["x"], outputs=["y"])
        score = _make_score(
            name="mixed",
            tasks=[t1],
        )
        result = FakeSonataAnalysisResult(
            region_statuses={"r0": "static", "r1": "dynamic"},
        )
        c = build_schedule(score, result)

        assert len(c.regions) == 2
        assert c.regions[0].kind == "static"
        assert c.regions[0].region_id == "r0"
        assert c.regions[1].kind == "dynamic"
        assert c.regions[1].region_id == "r1"
        assert c.regions[1].dynamic_mode == "backend_dynamic"

    def test_boundary_between_static_and_dynamic(self):
        t1 = _task(0, func_id=1, core_type="aic", storage_keys=["x"], outputs=["intermediate"])
        t2 = _task(1, func_id=2, core_type="aic", storage_keys=["intermediate"], outputs=["y"])
        score = _make_score(
            name="boundary_test",
            tasks=[t1, t2],
            deps=[Dependency(producer=0, consumer=1)],
        )
        result = FakeSonataAnalysisResult(
            region_statuses={"r0": "static", "r1": "dynamic"},
        )
        c = build_schedule(score, result)

        assert len(c.boundaries) >= 1
        found = [b for b in c.boundaries if b.from_region == "r0" and b.to_region == "r1"]
        assert found, f"no boundary from r0→r1 in {c.boundaries}"

    def test_fallback_reason_propagated(self):
        score = _make_score(name="fallback")
        result = FakeSonataAnalysisResult(
            fallback_reasons=["unsupported_root_kind"],
        )
        c = build_schedule(score, result)
        assert c.fallback_policy is not None
        assert c.fallback_policy.value == "partial_fallback"

    def test_fingerprint_computed(self):
        t1 = _task(0, func_id=1, core_type="aic", storage_keys=["x"])
        score_a = _make_score(name="fp_test", tasks=[t1])
        score_b = _make_score(name="fp_test", tasks=[t1])
        result = FakeSonataAnalysisResult(region_statuses={"r0": "static"})

        c1 = build_schedule(score_a, result)
        c2 = build_schedule(score_b, result)
        assert c1.fingerprint == c2.fingerprint

    def test_schema_version_default(self):
        score = _make_score()
        result = FakeSonataAnalysisResult()
        c = build_schedule(score, result)
        assert c.schema_version == SONATA_SCHEDULE_SCHEMA_VERSION
        assert c.runtime_contract == "sonata_schedule_v2"

    def test_arg_identity_fallback_when_no_storage_keys(self):
        t1 = _task(0, func_id=1, core_type="aic", args=[1.0, 2.0])
        score = _make_score(name="no_keys", tasks=[t1])
        result = FakeSonataAnalysisResult(region_statuses={"r0": "static"})
        c = build_schedule(score, result)

        scheduled = c.regions[0].tasks[0]
        assert len(scheduled.args) == 2
        assert scheduled.args[0].arg_identity.startswith("0:arg")

    def test_kernel_identity_fallback_when_no_name(self):
        t1 = _task(0, func_id=1, core_type="aic")
        score = _make_score(name="no_name", tasks=[t1])
        result = FakeSonataAnalysisResult(region_statuses={"r0": "static"})
        c = build_schedule(score, result)

        assert c.regions[0].tasks[0].kernel_identity == "task_0"

    def test_schedule_json_round_trip(self):
        t1 = _task(0, func_id=1, core_type="aic", storage_keys=["x"], outputs=["y"], name="load")
        score = _make_score(name="roundtrip", tasks=[t1])
        result = FakeSonataAnalysisResult(region_statuses={"r0": "static"})
        c1 = build_schedule(score, result)

        j = c1.to_json()
        c2 = SonataScheduleContract.from_json(j)

        assert c2.fingerprint == c1.fingerprint
        assert len(c2.regions) == len(c1.regions)
        assert c2.regions[0].tasks[0].kernel_identity == "load"
        assert c2.regions[0].tasks[0].name == "load"
