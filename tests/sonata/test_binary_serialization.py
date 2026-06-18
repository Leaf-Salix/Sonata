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


def _make_static_region(tasks=(), deps=()):
    return ScheduledRegion(region_id="r0", kind="static", tasks=tuple(tasks), deps=tuple(deps))


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

    def test_all_directions_round_trip_values(self):
        """Verify each direction round-trips to its correct value, not just count."""
        args = [
            ArgBinding(arg_identity="a", direction=ArgDirection.INPUT),
            ArgBinding(arg_identity="b", direction=ArgDirection.OUTPUT),
            ArgBinding(arg_identity="c", direction=ArgDirection.INOUT),
            ArgBinding(arg_identity="d", direction=ArgDirection.OUTPUT_EXISTING),
            ArgBinding(arg_identity="e", direction=ArgDirection.NO_DEP),
            ArgBinding(arg_identity="n", direction=ArgDirection.SCALAR),
        ]
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic", args=tuple(args))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,))
        c = SonataScheduleContract(fingerprint="fp_dir", regions=(r0,))
        d = c.to_binary()
        c2 = SonataScheduleContract.from_binary(d)
        rtasks = c2.regions[0].tasks[0].args
        dirs = [a.direction for a in rtasks]
        assert dirs[0] == ArgDirection.INPUT
        assert dirs[1] == ArgDirection.OUTPUT
        assert dirs[2] == ArgDirection.INOUT
        assert dirs[3] == ArgDirection.OUTPUT_EXISTING
        assert dirs[4] == ArgDirection.NO_DEP
        assert dirs[5] == ArgDirection.SCALAR

    def test_scope_mode_round_trip(self):
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic",
            args=(ArgBinding(arg_identity="x"),))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,), scope_mode=ScopeMode.MANUAL)
        c = SonataScheduleContract(fingerprint="fp_scope", regions=(r0,))
        d = c.to_binary()
        c2 = SonataScheduleContract.from_binary(d)
        assert c2.regions[0].scope_mode == ScopeMode.MANUAL

    def test_aiv_core_type(self):
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aiv",
            args=(ArgBinding(arg_identity="x"),))
        r0 = _make_static_region(tasks=(t1,))
        c = SonataScheduleContract(fingerprint="fp_aiv", regions=(r0,))
        d = c.to_binary()
        c2 = SonataScheduleContract.from_binary(d)
        assert c2.regions[0].tasks[0].core_type == "aiv"

    def test_bad_magic_rejected(self):
        # Header is 88 bytes; provide enough for "too short" not to trigger first
        data = b"\x00" * 88
        with pytest.raises(ValueError, match="bad magic"):
            SonataScheduleContract.from_binary(data)

    def test_truncated_data_rejected(self):
        with pytest.raises(ValueError, match="too short"):
            SonataScheduleContract.from_binary(b"\x00" * 10)

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

    def test_multi_region_deps_round_trip(self):
        """S4: Deps in each region round-trip independently."""
        t0 = ScheduledTask(task_id=0, kernel_identity="a", func_id=1, core_type="aic",
            args=(ArgBinding(arg_identity="x"),))
        t1 = ScheduledTask(task_id=1, kernel_identity="b", func_id=2, core_type="aic",
            args=(ArgBinding(arg_identity="y"),))
        t2 = ScheduledTask(task_id=2, kernel_identity="c", func_id=3, core_type="aic",
            args=(ArgBinding(arg_identity="z"),))
        t3 = ScheduledTask(task_id=3, kernel_identity="d", func_id=4, core_type="aic",
            args=(ArgBinding(arg_identity="w"),))
        r0 = ScheduledRegion(region_id="r0", kind="static",
            tasks=(t0, t1), deps=(ScheduleDep(producer=0, consumer=1),))
        r1 = ScheduledRegion(region_id="r1", kind="static",
            tasks=(t2, t3), deps=(ScheduleDep(producer=0, consumer=1),))
        c = SonataScheduleContract(fingerprint="fp_mr", regions=(r0, r1))
        data = c.to_binary()
        c2 = SonataScheduleContract.from_binary(data)
        assert len(c2.regions) == 2
        assert len(c2.regions[0].deps) == 1
        assert len(c2.regions[1].deps) == 1
        assert c2.regions[0].deps[0] == ScheduleDep(producer=0, consumer=1)
        assert c2.regions[1].deps[0] == ScheduleDep(producer=0, consumer=1)
        assert c2.regions[0].tasks[0].func_id == 1
        assert c2.regions[1].tasks[0].func_id == 3

    def test_runtime_slot_round_trip(self):
        """S5: runtime_slot=0 survives round-trip (not conflated with None)."""
        a_slot0 = ArgBinding(arg_identity="a", runtime_slot=0, direction=ArgDirection.INPUT)
        a_none = ArgBinding(arg_identity="b", runtime_slot=None, direction=ArgDirection.OUTPUT)
        a_slot5 = ArgBinding(arg_identity="c", runtime_slot=5, direction=ArgDirection.INOUT)
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic",
            args=(a_slot0, a_none, a_slot5))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,))
        c = SonataScheduleContract(fingerprint="fp_slot", regions=(r0,))
        d = c.to_binary()
        c2 = SonataScheduleContract.from_binary(d)
        args = c2.regions[0].tasks[0].args
        assert args[0].runtime_slot == 0
        assert args[1].runtime_slot is None
        assert args[2].runtime_slot == 5

    def test_string_table_kernel_identity_round_trip(self):
        """W4: kernel_identity and arg_identity survive binary round-trip via string table."""
        a1 = ArgBinding(arg_identity="input_tensor_x", runtime_slot=0, direction=ArgDirection.INPUT)
        a2 = ArgBinding(arg_identity="output_tensor_z", runtime_slot=1, direction=ArgDirection.OUTPUT)
        t1 = ScheduledTask(task_id=0, kernel_identity="tile_abs", func_id=3, core_type="aic",
            args=(a1, a2))
        t2 = ScheduledTask(task_id=1, kernel_identity="tile_add", func_id=5, core_type="aic",
            args=(ArgBinding(arg_identity="scratch_buf"),))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1, t2),
            deps=(ScheduleDep(producer=0, consumer=1),))
        c = SonataScheduleContract(fingerprint="fp_str", regions=(r0,))
        d = c.to_binary()
        c2 = SonataScheduleContract.from_binary(d)
        assert c2.regions[0].tasks[0].kernel_identity == "tile_abs"
        assert c2.regions[0].tasks[1].kernel_identity == "tile_add"
        assert c2.regions[0].tasks[0].args[0].arg_identity == "input_tensor_x"
        assert c2.regions[0].tasks[0].args[1].arg_identity == "output_tensor_z"
        assert c2.regions[0].tasks[1].args[0].arg_identity == "scratch_buf"
