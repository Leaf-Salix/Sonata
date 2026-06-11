"""Tests for HBGScheduleBackend — topology correctness validation."""

import pytest

from sonata.schedule import (
    ArgBinding,
    ScheduleDep,
    ScheduledRegion,
    ScheduledTask,
    SonataScheduleContract,
)
from sonata.backends.hbg_backend import HBGScheduleBackend, HBGScheduleResult


class TestHBGScheduleBackend:
    def test_consume_empty_schedule(self):
        backend = HBGScheduleBackend()
        result = backend.consume(SonataScheduleContract())
        assert result.success
        assert result.tasks == 0
        assert result.edges == 0

    def test_consume_static_region(self):
        t1 = ScheduledTask(task_id=0, kernel_identity="add", func_id=1, core_type="aic",
            args=(ArgBinding(arg_identity="x"), ArgBinding(arg_identity="y")), outputs=("z",))
        t2 = ScheduledTask(task_id=1, kernel_identity="mul", func_id=2, core_type="aic",
            args=(ArgBinding(arg_identity="z"),), outputs=("w",))
        r = ScheduledRegion(
            region_id="r0", kind="static",
            tasks=(t1, t2),
            deps=(ScheduleDep(producer=0, consumer=1),),
        )
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))
        result = HBGScheduleBackend().consume(c)
        assert result.success
        assert result.tasks == 2
        assert result.edges == 1
        assert result.plan is not None
        assert result.plan.task_count() == 2
        assert result.plan.edge_count() == 1

    def test_consume_dynamic_only(self):
        r = ScheduledRegion(region_id="r0", kind="dynamic", mode="backend_dynamic")
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))
        result = HBGScheduleBackend().consume(c)
        assert result.success
        assert result.tasks == 0
        assert result.edges == 0

    def test_consume_mixed_regions(self):
        t = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic", outputs=("y",))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t,))
        r1 = ScheduledRegion(region_id="r1", kind="dynamic", mode="backend_dynamic")
        c = SonataScheduleContract(fingerprint="fp", regions=(r0, r1))
        result = HBGScheduleBackend().consume(c)
        assert result.success
        assert result.tasks == 1
        assert result.edges == 0

    def test_rejects_invalid_dep(self):
        r = ScheduledRegion(
            region_id="r", kind="static",
            tasks=(ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic"),),
            deps=(ScheduleDep(producer=99, consumer=0),),
        )
        c = SonataScheduleContract(fingerprint="f", regions=(r,))
        result = HBGScheduleBackend().consume(c)
        assert not result.success
        assert len(result.reasons) >= 1
        assert result.plan is None

    def test_rejects_self_edge(self):
        r = ScheduledRegion(
            region_id="r", kind="static",
            tasks=(ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic"),),
            deps=(ScheduleDep(producer=0, consumer=0),),
        )
        c = SonataScheduleContract(fingerprint="f", regions=(r,))
        result = HBGScheduleBackend().consume(c)
        assert not result.success

    def test_topology_task_count_matches(self):
        t1 = ScheduledTask(task_id=0, kernel_identity="a", func_id=10, core_type="aic")
        t2 = ScheduledTask(task_id=1, kernel_identity="b", func_id=11, core_type="aiv")
        t3 = ScheduledTask(task_id=2, kernel_identity="c", func_id=12, core_type="aic")
        r = ScheduledRegion(
            region_id="r", kind="static",
            tasks=(t1, t2, t3),
            deps=(
                ScheduleDep(producer=0, consumer=1),
                ScheduleDep(producer=0, consumer=2),
            ),
        )
        c = SonataScheduleContract(fingerprint="f", regions=(r,))
        result = HBGScheduleBackend().consume(c)
        assert result.success
        assert result.tasks == 3
        assert result.edges == 2
