"""Tests for TMARB mapping module — validator, trace, pseudocode."""

import pytest

from sonata.schedule import (
    ArgBinding,
    ArgDirection,
    RegionBoundary,
    ScheduleDep,
    ScheduledRegion,
    ScheduledTask,
    ScopeMode,
    SonataScheduleContract,
)
from sonata.mapping.validator import validate_tmarb_mapping
from sonata.mapping.trace import generate_trace, trace_to_json
from sonata.mapping.pseudocode import generate_pseudocode


def _make_static_region(region_id="r0", tasks=None, deps=None):
    return ScheduledRegion(
        region_id=region_id,
        kind="static",
        tasks=tasks or (),
        deps=deps or (),
    )


def _task(task_id=0, kernel_identity="k", func_id=1, core_type="aic", args=None, outputs=None, name=None):
    return ScheduledTask(
        task_id=task_id,
        kernel_identity=kernel_identity,
        func_id=func_id,
        core_type=core_type,
        args=tuple(args or [ArgBinding(arg_identity="x")]),
        outputs=tuple(outputs or []),
        name=name,
    )


class TestTraceGeneration:
    def test_empty_schedule_trace(self):
        c = SonataScheduleContract(fingerprint="fp")
        trace = generate_trace(c)
        assert isinstance(trace, list)

    def test_static_region_trace(self):
        t = _task(0, "add", func_id=3)
        r = _make_static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))
        trace = generate_trace(c)
        assert len(trace) >= 2  # PTO2_SCOPE + add_input + submit
        assert any(e.api == "rt_submit_aic_task" for e in trace)

    def test_dynamic_region_trace(self):
        r = ScheduledRegion(region_id="r0", kind="dynamic", dynamic_mode="backend_dynamic")
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))
        trace = generate_trace(c)
        assert len(trace) >= 1
        assert trace[0].api == "PTO2_SCOPE"

    def test_trace_deterministic(self):
        t = _task(0, "add")
        r = _make_static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))
        t1 = generate_trace(c)
        t2 = generate_trace(c)
        assert len(t1) == len(t2)
        for e1, e2 in zip(t1, t2):
            assert e1.api == e2.api

    def test_trace_direction_normalization(self):
        t = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic",
            args=(ArgBinding(arg_identity="x", direction=ArgDirection.OUTPUT),))
        r = _make_static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))
        trace = generate_trace(c)
        assert any(e.api == "add_output" for e in trace)


class TestPseudocodeGeneration:
    def test_empty_schedule_pseudocode(self):
        c = SonataScheduleContract(fingerprint="fp")
        code = generate_pseudocode(c)
        assert "extern \"C\" void aicpu_orchestration_entry" in code
        assert "rt_orchestration_done" in code

    def test_static_region_pseudocode(self):
        t = _task(0, "add", func_id=3)
        r = _make_static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))
        code = generate_pseudocode(c)
        assert "PTO2_SCOPE()" in code
        assert "rt_submit_aic_task" in code

    def test_dynamic_region_pseudocode(self):
        r = ScheduledRegion(region_id="r0", kind="dynamic", dynamic_mode="backend_dynamic")
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))
        code = generate_pseudocode(c)
        assert "PTO2_SCOPE()" in code

    def test_mixed_kernel_pseudocode(self):
        from sonata.schedule import MixedKernels
        mk = MixedKernels(aic_func_id=3, aiv_func_id=5)
        t = ScheduledTask(task_id=0, kernel_identity="matmul", func_id=3, core_type="mixed",
            mixed_kernels=mk, args=(ArgBinding(arg_identity="x"),))
        r = _make_static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))
        code = generate_pseudocode(c)
        assert "rt_submit_task" in code

    def test_all_six_directions(self):
        args = [
            ArgBinding(arg_identity="a", direction=ArgDirection.INPUT),
            ArgBinding(arg_identity="b", direction=ArgDirection.OUTPUT),
            ArgBinding(arg_identity="c", direction=ArgDirection.INOUT),
            ArgBinding(arg_identity="d", direction=ArgDirection.OUTPUT_EXISTING),
            ArgBinding(arg_identity="e", direction=ArgDirection.NO_DEP),
            ArgBinding(arg_identity="n", direction=ArgDirection.SCALAR),
        ]
        t = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic", args=tuple(args))
        r = _make_static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))
        code = generate_pseudocode(c)
        assert "add_input" in code
        assert "add_output" in code
        assert "add_inout" in code
        assert "add_no_dep" in code
        assert "add_scalar" in code


class TestMappingValidator:
    def test_valid_schedule_no_errors(self):
        t = _task(0, "add")
        r = _make_static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))
        errors = validate_tmarb_mapping(c)
        assert errors == ()

    def test_empty_schedule_no_errors(self):
        c = SonataScheduleContract(fingerprint="fp")
        errors = validate_tmarb_mapping(c)
        assert errors == ()

    def test_missing_mixed_kernels(self):
        t = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="mixed")
        r = _make_static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))
        errors = validate_tmarb_mapping(c)
        assert len(errors) == 1
        assert errors[0].code == "schedule_bad_nullable"

    def test_unbound_func_id_warning(self):
        t = ScheduledTask(task_id=0, kernel_identity="k", func_id=None, core_type="aic")
        r = _make_static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))
        errors = validate_tmarb_mapping(c)
        assert len(errors) == 1
        assert errors[0].code == "binding_func_id_not_found"

    def test_auto_inside_manual_rejected(self):
        r0 = ScheduledRegion(region_id="r0", kind="static", scope_mode=ScopeMode.MANUAL,
            tasks=(ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic"),))
        r1 = ScheduledRegion(region_id="r1", kind="static", scope_mode=ScopeMode.AUTO,
            tasks=(ScheduledTask(task_id=1, kernel_identity="k", func_id=2, core_type="aic"),))
        c = SonataScheduleContract(fingerprint="fp", regions=(r0, r1))
        errors = validate_tmarb_mapping(c)
        assert len(errors) == 1
        assert errors[0].code == "schedule_dynamic_dep"

    def test_manual_then_dynamic_is_ok(self):
        r0 = ScheduledRegion(region_id="r0", kind="static", scope_mode=ScopeMode.MANUAL,
            tasks=(ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic"),))
        r1 = ScheduledRegion(region_id="r1", kind="dynamic", dynamic_mode="backend_dynamic")
        c = SonataScheduleContract(fingerprint="fp", regions=(r0, r1))
        errors = validate_tmarb_mapping(c)
        assert errors == ()

    def test_output_existing_in_pseudocode(self):
        args = [ArgBinding(arg_identity="x", direction=ArgDirection.OUTPUT_EXISTING)]
        t = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic", args=tuple(args))
        r = _make_static_region(tasks=(t,))
        c = SonataScheduleContract(fingerprint="fp", regions=(r,))
        code = generate_pseudocode(c)
        assert "add_output" in code
        assert "x" in code
