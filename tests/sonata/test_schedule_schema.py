"""Tests for SonataScheduleContract schema — construction, serialization, round-trip."""

import json
from pathlib import Path
import pytest

from sonata.schedule import (
    ArgBinding,
    RegionBoundary,
    ScheduleDep,
    ScheduleGuard,
    ScheduledRegion,
    ScheduledTask,
    SonataScheduleContract,
    SONATA_SCHEDULE_SCHEMA_VERSION,
    RUNTIME_CONTRACT,
)


def make_minimal_contract() -> SonataScheduleContract:
    return SonataScheduleContract(
        fingerprint="fp_test_001",
        regions=_make_two_regions(),
        boundaries=(RegionBoundary(from_region="r0", to_region="r1", tensors=("y",)),),
        guards=(ScheduleGuard(guard_id="g0", kind="shape_range", severity="hard",
                symbolic_name="x", min_value=64, max_value=128),),
    )


def _make_two_regions():
    t1 = ScheduledTask(
        task_id=0,
        kernel_identity="load_tile",
        func_id=3,
        core_type="aic",
        args=(ArgBinding(arg_identity="x"), ArgBinding(arg_identity="tmp")),
        outputs=("tmp",),
    )
    t2 = ScheduledTask(
        task_id=1,
        kernel_identity="store_tile",
        func_id=5,
        core_type="aiv",
        args=(ArgBinding(arg_identity="tmp"), ArgBinding(arg_identity="y")),
        outputs=("y",),
    )
    r0 = ScheduledRegion(
        region_id="r0",
        kind="static",
        tasks=(t1, t2),
        deps=(ScheduleDep(producer=0, consumer=1),),
    )
    r1 = ScheduledRegion(region_id="r1", kind="dynamic", mode="backend_dynamic")
    return (r0, r1)


class TestSonataScheduleContract:
    def test_construction_defaults(self):
        c = SonataScheduleContract()
        assert c.schema_version == SONATA_SCHEDULE_SCHEMA_VERSION
        assert c.runtime_contract == RUNTIME_CONTRACT
        assert c.fingerprint == ""
        assert c.regions == ()
        assert c.boundaries == ()
        assert c.guards == ()

    def test_construction_minimal(self):
        c = make_minimal_contract()
        assert c.schema_version == 1
        assert c.fingerprint == "fp_test_001"
        assert len(c.regions) == 2
        assert c.regions[0].kind == "static"
        assert c.regions[1].kind == "dynamic"

    def test_to_dict_shape(self):
        c = make_minimal_contract()
        d = c.to_dict()
        assert d["schema_version"] == 1
        assert d["runtime_contract"] == RUNTIME_CONTRACT
        assert d["fingerprint"] == "fp_test_001"
        assert len(d["regions"]) == 2
        assert d["regions"][0]["kind"] == "static"
        assert d["regions"][1]["kind"] == "dynamic"
        assert d["regions"][1]["mode"] == "backend_dynamic"

    def test_round_trip_dict(self):
        c = make_minimal_contract()
        c2 = SonataScheduleContract.from_dict(c.to_dict())
        assert c2.fingerprint == c.fingerprint
        assert len(c2.regions) == len(c.regions)
        assert c2.regions[0].kind == c.regions[0].kind
        assert len(c2.regions[0].tasks) == len(c.regions[0].tasks)
        assert c2.regions[0].tasks[0].kernel_identity == c.regions[0].tasks[0].kernel_identity
        assert c2.regions[0].tasks[0].func_id == c.regions[0].tasks[0].func_id

    def test_round_trip_json(self):
        c = make_minimal_contract()
        j = c.to_json()
        c2 = SonataScheduleContract.from_json(j)
        assert c2.fingerprint == c.fingerprint
        assert len(c2.regions) == 2
        assert c2.regions[0].tasks[0].kernel_identity == "load_tile"

    def test_write_read_json(self, tmp_path):
        c = make_minimal_contract()
        path = tmp_path / "schedule.json"
        c.write_json(path)
        assert path.exists()
        c2 = SonataScheduleContract.read_json(path)
        assert c2.fingerprint == c.fingerprint

    def test_nullable_func_id_preserved(self):
        t = ScheduledTask(task_id=0, kernel_identity="k", func_id=None, core_type="aic")
        r = ScheduledRegion(region_id="r", kind="static", tasks=(t,))
        c = SonataScheduleContract(fingerprint="f", regions=(r,))
        d = c.to_dict()
        assert d["regions"][0]["tasks"][0]["func_id"] is None
        c2 = SonataScheduleContract.from_dict(d)
        assert c2.regions[0].tasks[0].func_id is None

    def test_nullable_runtime_slot_preserved(self):
        t = ScheduledTask(
            task_id=0, kernel_identity="k", func_id=1, core_type="aic",
            args=(ArgBinding(arg_identity="x", runtime_slot=None),),
        )
        r = ScheduledRegion(region_id="r", kind="static", tasks=(t,))
        c = SonataScheduleContract(fingerprint="f", regions=(r,))
        d = c.to_dict()
        assert d["regions"][0]["tasks"][0]["args"][0]["runtime_slot"] is None

    def test_schema_version_validation(self):
        c = make_minimal_contract()
        d = c.to_dict()
        d["schema_version"] = 999
        c2 = SonataScheduleContract.from_dict(d)
        assert c2.schema_version == 999


class TestScheduledRegion:
    def test_static_region_has_tasks_and_deps(self):
        t = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic")
        r = ScheduledRegion(
            region_id="r", kind="static",
            tasks=(t,),
            deps=(ScheduleDep(producer=0, consumer=0),),
        )
        assert r.kind == "static"
        assert len(r.tasks) == 1
        assert len(r.deps) == 1

    def test_dynamic_region_has_mode(self):
        r = ScheduledRegion(region_id="r", kind="dynamic", mode="backend_dynamic")
        assert r.kind == "dynamic"
        assert r.mode == "backend_dynamic"
        assert r.tasks == ()
        assert r.deps == ()


class TestArgBinding:
    def test_basic_binding(self):
        a = ArgBinding(arg_identity="x", runtime_slot=3)
        assert a.arg_identity == "x"
        assert a.runtime_slot == 3

    def test_nullable_slot(self):
        a = ArgBinding(arg_identity="x")
        assert a.arg_identity == "x"
        assert a.runtime_slot is None


class TestGoldenFixtures:
    GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

    def test_basic_static_dynamic(self):
        path = self.GOLDEN_DIR / "basic_static_dynamic.json"
        assert path.exists(), f"golden fixture not found: {path}"
        c = SonataScheduleContract.read_json(path)
        assert c.schema_version == 1
        assert c.runtime_contract == "sonata_schedule_v1"
        assert c.fingerprint == "golden_fp_001"
        assert len(c.regions) == 2
        assert c.regions[0].kind == "static"
        assert c.regions[1].kind == "dynamic"
        assert c.regions[1].mode == "backend_dynamic"
        assert len(c.boundaries) == 1
        assert c.boundaries[0].from_region == "r0"
        assert c.boundaries[0].to_region == "r1"

    def test_golden_files_exist(self):
        files = list(self.GOLDEN_DIR.glob("*.json"))
        assert len(files) >= 1, f"no golden fixtures in {self.GOLDEN_DIR}"
