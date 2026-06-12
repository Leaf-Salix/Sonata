"""Tests for binary flat schedule serialization (from/to sonata_tmarb flat format)."""

import pytest

from sonata.schedule import (
    ArgBinding,
    ArgDirection,
    ScheduleDep,
    ScheduledRegion,
    ScheduledTask,
    ScopeMode,
    SonataScheduleContract,
)


class TestBinarySerialization:
    def test_empty_schedule_binary(self):
        c = SonataScheduleContract(fingerprint="fp")
        data = c.to_binary()
        c2 = SonataScheduleContract.from_binary(data)
        assert c2.fingerprint == "fp"
        assert len(c2.regions) == 0

    def test_static_only_round_trip(self):
        t1 = ScheduledTask(task_id=0, kernel_identity="add", func_id=3, core_type="aic",
            args=(ArgBinding(arg_identity="x", direction=ArgDirection.INPUT),))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,),
            scope_mode=ScopeMode.AUTO)
        c = SonataScheduleContract(fingerprint="fp_001", regions=(r0,))
        data = c.to_binary()
        c2 = SonataScheduleContract.from_binary(data)
        assert c2.fingerprint == "fp_001"
        assert len(c2.regions) == 1
        assert c2.regions[0].kind == "static"
        assert len(c2.regions[0].tasks) == 1

    def test_static_with_deps_round_trip(self):
        t1 = ScheduledTask(task_id=0, kernel_identity="add", func_id=3, core_type="aic",
            args=(ArgBinding(arg_identity="x"),))
        t2 = ScheduledTask(task_id=1, kernel_identity="mul", func_id=5, core_type="aic",
            args=(ArgBinding(arg_identity="z"),))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1, t2),
            deps=(ScheduleDep(producer=0, consumer=1),))
        c = SonataScheduleContract(fingerprint="fp_deps", regions=(r0,))
        data = c.to_binary()
        c2 = SonataScheduleContract.from_binary(data)
        assert len(c2.regions) == 1
        assert len(c2.regions[0].tasks) == 2
        assert len(c2.regions[0].deps) == 1

    def test_static_dynamic_mixed(self):
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic",
            args=(ArgBinding(arg_identity="x"),))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,))
        r1 = ScheduledRegion(region_id="r1", kind="dynamic", dynamic_mode="backend_dynamic")
        c = SonataScheduleContract(fingerprint="fp_mix", regions=(r0, r1))
        data = c.to_binary()
        c2 = SonataScheduleContract.from_binary(data)
        assert len(c2.regions) == 2
        assert c2.regions[0].kind == "static"
        assert c2.regions[1].kind == "dynamic"

    def test_all_directions_in_args(self):
        args = [
            ArgBinding(arg_identity="a", direction=ArgDirection.INPUT),
            ArgBinding(arg_identity="b", direction=ArgDirection.OUTPUT),
            ArgBinding(arg_identity="c", direction=ArgDirection.INOUT),
            ArgBinding(arg_identity="d", direction=ArgDirection.OUTPUT_EXISTING),
            ArgBinding(arg_identity="e", direction=ArgDirection.NO_DEP),
            ArgBinding(arg_identity="n", direction=ArgDirection.SCALAR),
        ]
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic",
            args=tuple(args))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,))
        c = SonataScheduleContract(fingerprint="fp_dir", regions=(r0,))
        data = c.to_binary()
        c2 = SonataScheduleContract.from_binary(data)
        assert len(c2.regions[0].tasks[0].args) == 6

    def test_magic_number(self):
        c = SonataScheduleContract(fingerprint="fp")
        data = c.to_binary()
        import struct
        magic = struct.unpack_from("<i", data, 0)[0]
        assert magic == 0x534F4E41  # "SONA"

    def test_deterministic(self):
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic",
            args=(ArgBinding(arg_identity="x"),))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,))
        c = SonataScheduleContract(fingerprint="fp_det", regions=(r0,))
        d1 = c.to_binary()
        d2 = c.to_binary()
        assert d1 == d2

    def test_large_schedule(self):
        tasks = tuple(
            ScheduledTask(task_id=i, kernel_identity=f"k{i}", func_id=i, core_type="aic",
                args=(ArgBinding(arg_identity=f"x{i}"),))
            for i in range(100)
        )
        deps = tuple(
            ScheduleDep(producer=i, consumer=i + 1)
            for i in range(99)
        )
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=tasks, deps=deps)
        c = SonataScheduleContract(fingerprint="fp_large", regions=(r0,))
        data = c.to_binary()
        assert len(data) > 0
        c2 = SonataScheduleContract.from_binary(data)
        assert len(c2.regions[0].tasks) == 100
        assert len(c2.regions[0].deps) == 99
