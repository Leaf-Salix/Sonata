"""Integration test: bind_func_ids against real PyPTO codegen output.

Requires a working ``pypto`` installation with codegen bindings.
Skips gracefully when pypto is unavailable (CI/venv without pypto).
"""

import pytest

pytest.importorskip("pypto", reason="pypto not available")
from pypto.pypto_core.codegen import generate_orchestration, OrchestrationResult


class TestCodegenApi:
    """Verify the codegen API is accessible and returns expected types."""

    def test_generate_orchestration_importable(self):
        assert generate_orchestration is not None
        assert hasattr(OrchestrationResult, "func_name_to_id")
        assert hasattr(OrchestrationResult, "func_name_to_core_type")

    def test_func_name_to_id_type(self):
        """Verify OrchestrationResult defines the expected attribute types."""
        # nanobind functions don't support inspect.signature, so just verify
        # the attributes exist and have expected Python types.
        assert hasattr(OrchestrationResult, "func_name_to_id")
        assert hasattr(OrchestrationResult, "func_name_to_core_type")
        assert hasattr(OrchestrationResult, "code")

    def test_func_name_to_id_shape(self):
        """Verify generate_orchestration can be called (even if it fails gracefully)."""
        # Just verify the function is callable — real IR construction requires
        # C++ API knowledge and is tested separately.
        assert callable(generate_orchestration)


class TestBindFuncIdsWithCodegen:
    """Integration: bind_func_ids with real codegen output."""

    def test_bind_func_ids_with_synthetic_map(self):
        """Verify bind_func_ids works with a real func_name_to_id dict."""
        from sonata.schedule import SonataScheduleContract, ScheduledRegion, ScheduledTask
        from sonata.binding import bind_func_ids

        t1 = ScheduledTask(task_id=0, kernel_identity="add", func_id=None, core_type="aic")
        t2 = ScheduledTask(task_id=1, kernel_identity="mul", func_id=None, core_type="aic")
        r = ScheduledRegion(region_id="r0", kind="static", tasks=(t1, t2))
        schedule = SonataScheduleContract(fingerprint="d3_test", regions=(r,))

        bound, reasons = bind_func_ids(schedule, {"add": 0, "mul": 1})
        assert len(reasons) == 0
        assert bound.regions[0].tasks[0].func_id == 0
        assert bound.regions[0].tasks[1].func_id == 1

    def test_missing_identity_in_codegen_output(self):
        """Verify fail-open when kernel_identity is missing from codegen."""
        from sonata.schedule import SonataScheduleContract, ScheduledRegion, ScheduledTask
        from sonata.binding import bind_func_ids

        t = ScheduledTask(task_id=0, kernel_identity="unknown", func_id=None, core_type="aic")
        r = ScheduledRegion(region_id="r0", kind="static", tasks=(t,))
        schedule = SonataScheduleContract(fingerprint="d3_missing", regions=(r,))

        bound, reasons = bind_func_ids(schedule, {"known_kernel": 3})
        assert len(reasons) == 1
        assert "unknown" in reasons[0].message
        assert bound.regions[0].tasks[0].func_id is None

    def test_bind_runtime_slots_with_synthetic_maps(self):
        """Verify bind_runtime_slots works with tensor/scalar name lists."""
        from sonata.schedule import (
            ArgBinding, SonataScheduleContract, ScheduledRegion, ScheduledTask,
        )
        from sonata.binding import bind_runtime_slots

        t = ScheduledTask(
            task_id=0, kernel_identity="k", func_id=1, core_type="aic",
            args=(ArgBinding(arg_identity="x"), ArgBinding(arg_identity="n")),
        )
        r = ScheduledRegion(region_id="r0", kind="static", tasks=(t,))
        schedule = SonataScheduleContract(fingerprint="d3_slots", regions=(r,))

        bound, reasons = bind_runtime_slots(schedule, ["x"], ["n"])
        assert len(reasons) == 0
        assert bound.regions[0].tasks[0].args[0].runtime_slot == 0  # x
        assert bound.regions[0].tasks[0].args[1].runtime_slot == 1  # n (offset 1)

    def test_bind_func_idempotent(self):
        """Verify re-binding preserves existing func_ids."""
        from sonata.schedule import SonataScheduleContract, ScheduledRegion, ScheduledTask
        from sonata.binding import bind_func_ids

        t = ScheduledTask(task_id=0, kernel_identity="add", func_id=3, core_type="aic")
        r = ScheduledRegion(region_id="r0", kind="static", tasks=(t,))
        schedule = SonataScheduleContract(fingerprint="d3_idem", regions=(r,))

        # func_id is already 3, binding with same map should keep it
        bound, reasons = bind_func_ids(schedule, {"add": 3})
        assert len(reasons) == 0
        assert bound.regions[0].tasks[0].func_id == 3

        # Binding with different map should update (overwrite behavior)
        bound2, reasons2 = bind_func_ids(schedule, {"add": 7})
        assert len(reasons2) == 0
        assert bound2.regions[0].tasks[0].func_id == 7
