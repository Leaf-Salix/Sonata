"""Tests for Sonata schedule binding — func_id and runtime_slot binding."""

import pytest

from sonata.schedule import (
    ArgBinding,
    ScheduleDep,
    ScheduledRegion,
    ScheduledTask,
    SonataScheduleContract,
)
from sonata.binding import bind_func_ids, bind_runtime_slots


def _static_region(region_id="r0", tasks=None, deps=None):
    return ScheduledRegion(
        region_id=region_id,
        kind="static",
        tasks=tasks or (),
        deps=deps or (),
    )


def _task(task_id, kernel_identity="k", core_type="aic", args=None, outputs=None, name=None):
    return ScheduledTask(
        task_id=task_id,
        kernel_identity=kernel_identity,
        func_id=None,
        core_type=core_type,
        args=tuple(args or []),
        outputs=tuple(outputs or []),
        name=name,
    )


class TestBindFuncIds:
    def test_bind_single_task(self):
        t = _task(0, "add")
        r = _static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))

        result, reasons = bind_func_ids(c, {"add": 3})
        assert reasons == ()
        assert result.regions[0].tasks[0].func_id == 3

    def test_bind_multiple_tasks(self):
        t1 = _task(0, "add")
        t2 = _task(1, "mul")
        r = _static_region(tasks=(t1, t2))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))

        result, _ = bind_func_ids(c, {"add": 3, "mul": 5})
        assert result.regions[0].tasks[0].func_id == 3
        assert result.regions[0].tasks[1].func_id == 5

    def test_missing_identity_remains_none(self):
        t = _task(0, "add")
        r = _static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))

        result, reasons = bind_func_ids(c, {"mul": 5})
        assert result.regions[0].tasks[0].func_id is None
        assert len(reasons) == 1
        assert reasons[0].code == "binding_func_id_not_found"

    def test_overrides_take_priority(self):
        t = _task(0, "add")
        r = _static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))

        result, _ = bind_func_ids(c, {"add": 3}, overrides={"add": 99})
        assert result.regions[0].tasks[0].func_id == 99

    def test_empty_schedule(self):
        c = SonataScheduleContract()
        result, reasons = bind_func_ids(c, {})
        assert reasons == ()
        assert result is not None

    def test_dynamic_regions_skipped(self):
        r = ScheduledRegion(region_id="r0", kind="dynamic", mode="backend_dynamic")
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))

        result, reasons = bind_func_ids(c, {"add": 3})
        assert reasons == ()
        assert result.regions[0].kind == "dynamic"

    def test_no_reasons_on_full_match(self):
        t1 = _task(0, "a")
        t2 = _task(1, "b")
        r = _static_region(tasks=(t1, t2))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))

        _, reasons = bind_func_ids(c, {"a": 0, "b": 1})
        assert reasons == ()

    def test_same_kernel_identity_same_func_id(self):
        t1 = _task(0, "relu")
        t2 = _task(1, "relu")
        r = _static_region(tasks=(t1, t2))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))

        result, _ = bind_func_ids(c, {"relu": 7})
        assert result.regions[0].tasks[0].func_id == 7
        assert result.regions[0].tasks[1].func_id == 7

    def test_idempotent_second_call(self):
        t = _task(0, "add")
        r = _static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))

        c1, _ = bind_func_ids(c, {"add": 3})
        c2, _ = bind_func_ids(c1, {"add": 3})
        assert c2.regions[0].tasks[0].func_id == 3
        # Verify func_name_to_id always overwrites
        c3, _ = bind_func_ids(c1, {"add": 7})
        assert c3.regions[0].tasks[0].func_id == 7


class TestBindRuntimeSlots:
    def test_tensor_slot_assignment(self):
        t = _task(0, "k", args=[ArgBinding(arg_identity="x"), ArgBinding(arg_identity="y")])
        r = _static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))

        result, reasons = bind_runtime_slots(c, ["x", "y"], [])
        assert reasons == ()
        assert result.regions[0].tasks[0].args[0].runtime_slot == 0
        assert result.regions[0].tasks[0].args[1].runtime_slot == 1

    def test_scalar_slots_after_tensor_offset(self):
        t = _task(0, "k", args=[
            ArgBinding(arg_identity="x"),  # tensor
            ArgBinding(arg_identity="n"),  # scalar
        ])
        r = _static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))

        result, _ = bind_runtime_slots(c, ["x"], ["n"])
        assert result.regions[0].tasks[0].args[0].runtime_slot == 0  # x: tensor slot 0
        assert result.regions[0].tasks[0].args[1].runtime_slot == 1  # n: scalar offset 1

    def test_mixed_tensors_and_scalars(self):
        t = _task(0, "k", args=[
            ArgBinding(arg_identity="a"),  # tensor
            ArgBinding(arg_identity="b"),  # tensor
            ArgBinding(arg_identity="c"),  # scalar
            ArgBinding(arg_identity="d"),  # scalar
        ])
        r = _static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))

        result, _ = bind_runtime_slots(c, ["a", "b"], ["c", "d"])
        assert result.regions[0].tasks[0].args[0].runtime_slot == 0  # a
        assert result.regions[0].tasks[0].args[1].runtime_slot == 1  # b
        assert result.regions[0].tasks[0].args[2].runtime_slot == 2  # c (offset 2+0)
        assert result.regions[0].tasks[0].args[3].runtime_slot == 3  # d (offset 2+1)

    def test_missing_arg_identity(self):
        t = _task(0, "k", args=[ArgBinding(arg_identity="unknown")])
        r = _static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))

        result, reasons = bind_runtime_slots(c, [], [])
        assert len(reasons) == 1
        assert reasons[0].code == "binding_missing_slot"
        assert result.regions[0].tasks[0].args[0].runtime_slot is None

    def test_empty_args(self):
        t = _task(0, "k", args=[])
        r = _static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))

        result, reasons = bind_runtime_slots(c, [], [])
        assert reasons == ()
        assert result.regions[0].tasks[0].args == ()

    def test_dynamic_regions_not_affected(self):
        r = ScheduledRegion(region_id="r0", kind="dynamic", mode="backend_dynamic")
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))

        result, reasons = bind_runtime_slots(c, ["x"], [])
        assert reasons == ()
        assert result.regions[0].kind == "dynamic"
