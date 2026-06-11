"""Tests for schedule_validator — invariant checks before backend consumption."""

import pytest

from sonata.schedule import (
    ArgBinding,
    RegionBoundary,
    ScheduleDep,
    ScheduledRegion,
    ScheduledTask,
    SonataScheduleContract,
)
from sonata.schedule_validator import validate_schedule


def _static_region(region_id="r0", tasks=None, deps=None):
    return ScheduledRegion(
        region_id=region_id,
        kind="static",
        tasks=tasks or (
            ScheduledTask(task_id=0, kernel_identity="k0", func_id=1, core_type="aic"),
            ScheduledTask(task_id=1, kernel_identity="k1", func_id=2, core_type="aic"),
        ),
        deps=deps or (
            ScheduleDep(producer=0, consumer=1),
        ),
    )


def _dynamic_region(region_id="r1"):
    return ScheduledRegion(region_id=region_id, kind="dynamic", mode="backend_dynamic")


class TestValidateScheduleHappy:
    def test_empty_schedule_passes(self):
        errors = validate_schedule(SonataScheduleContract())
        assert errors == ()

    def test_single_static_region_passes(self):
        c = SonataScheduleContract(
            fingerprint="fp",
            regions=(_static_region(),),
        )
        assert validate_schedule(c) == ()

    def test_mixed_regions_passes(self):
        c = SonataScheduleContract(
            fingerprint="fp",
            regions=(_static_region("r0"), _dynamic_region("r1")),
        )
        assert validate_schedule(c) == ()

    def test_boundary_passes(self):
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic", outputs=("y",))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,))
        r1 = _dynamic_region("r1")
        c = SonataScheduleContract(
            fingerprint="fp",
            regions=(r0, r1),
            boundaries=(RegionBoundary(from_region="r0", to_region="r1", tensors=("y",)),),
        )
        assert validate_schedule(c) == ()


class TestValidateScheduleErrors:
    def test_dep_to_missing_producer(self):
        r = ScheduledRegion(
            region_id="r", kind="static",
            tasks=(ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic"),),
            deps=(ScheduleDep(producer=99, consumer=0),),
        )
        c = SonataScheduleContract(fingerprint="f", regions=(r,))
        errors = validate_schedule(c)
        assert len(errors) >= 1
        assert "producer" in errors[0].message.lower()

    def test_dep_to_missing_consumer(self):
        r = ScheduledRegion(
            region_id="r", kind="static",
            tasks=(ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic"),),
            deps=(ScheduleDep(producer=0, consumer=99),),
        )
        c = SonataScheduleContract(fingerprint="f", regions=(r,))
        errors = validate_schedule(c)
        assert len(errors) >= 1
        assert "consumer" in errors[0].message.lower()

    def test_boundary_missing_source_region(self):
        c = SonataScheduleContract(
            fingerprint="f",
            regions=(_static_region("r0"),),
            boundaries=(RegionBoundary(from_region="r99", to_region="r0", tensors=("x",)),),
        )
        errors = validate_schedule(c)
        assert len(errors) >= 1

    def test_boundary_missing_target_region(self):
        c = SonataScheduleContract(
            fingerprint="f",
            regions=(_static_region("r0"),),
            boundaries=(RegionBoundary(from_region="r0", to_region="r99", tensors=("x",)),),
        )
        errors = validate_schedule(c)
        assert len(errors) >= 1

    def test_boundary_tensor_not_produced(self):
        t = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic", outputs=("a",))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t,))
        r1 = _dynamic_region("r1")
        c = SonataScheduleContract(
            fingerprint="f",
            regions=(r0, r1),
            boundaries=(RegionBoundary(from_region="r0", to_region="r1", tensors=("z",)),),
        )
        errors = validate_schedule(c)
        assert len(errors) >= 1

    def test_fingerprint_mismatch(self):
        c = SonataScheduleContract(fingerprint="fp_a", regions=(_static_region(),))
        errors = validate_schedule(c, source_fingerprint="fp_b")
        assert len(errors) >= 1

    def test_negative_func_id(self):
        t = ScheduledTask(task_id=0, kernel_identity="k", func_id=-1, core_type="aic")
        r = ScheduledRegion(region_id="r", kind="static", tasks=(t,))
        c = SonataScheduleContract(fingerprint="f", regions=(r,))
        errors = validate_schedule(c)
        assert len(errors) >= 1

    def test_boundary_tensor_not_consumed(self):
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic", outputs=("y",))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,))
        t2 = ScheduledTask(task_id=1, kernel_identity="k", func_id=2, core_type="aic",
            args=(ArgBinding(arg_identity="z"),))
        r1 = ScheduledRegion(region_id="r1", kind="static", tasks=(t2,))
        c = SonataScheduleContract(
            fingerprint="f",
            regions=(r0, r1),
            boundaries=(RegionBoundary(from_region="r0", to_region="r1", tensors=("y",)),),
        )
        errors = validate_schedule(c)
        assert len(errors) >= 1

    def test_static_only_regions(self):
        t = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic")
        r = ScheduledRegion(region_id="r", kind="static", tasks=(t,))
        c = SonataScheduleContract(fingerprint="f", regions=(r,))
        assert validate_schedule(c) == ()

    def test_dynamic_only_regions(self):
        r = ScheduledRegion(region_id="r", kind="dynamic", mode="backend_dynamic")
        c = SonataScheduleContract(fingerprint="f", regions=(r,))
        assert validate_schedule(c) == ()

    def test_empty_static_region_without_tasks(self):
        r = ScheduledRegion(region_id="r", kind="static")
        c = SonataScheduleContract(fingerprint="f", regions=(r,))
        assert validate_schedule(c) == ()

    def test_no_deps(self):
        t = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic")
        r = ScheduledRegion(region_id="r", kind="static", tasks=(t,))
        c = SonataScheduleContract(fingerprint="f", regions=(r,))
        assert validate_schedule(c) == ()

    def test_three_mixed_regions(self):
        regions = [
            ScheduledRegion(region_id="r0", kind="static",
                tasks=(ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic"),)),
            ScheduledRegion(region_id="r1", kind="dynamic", mode="backend_dynamic"),
            ScheduledRegion(region_id="r2", kind="static",
                tasks=(ScheduledTask(task_id=1, kernel_identity="k", func_id=2, core_type="aic"),)),
        ]
        c = SonataScheduleContract(fingerprint="f", regions=tuple(regions))
        assert validate_schedule(c) == ()
