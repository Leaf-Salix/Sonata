# Copyright (c) PyPTO Contributors.
# ---------------------------------------------------------------------------
"""v0.28 Phase B: Multi-region end-to-end.

Creates a SonataScheduleContract with 2 regions (1 static + 1 dynamic),
serializes to binary, round-trips through from_binary, and runs on a2a3sim.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sonata.schedule import (
    ArgBinding,
    ArgDirection,
    ScheduleDep,
    ScheduledRegion,
    ScheduledTask,
    SonataScheduleContract,
)

PROJ_ROOT = Path(__file__).resolve().parents[2]

_HAS_A2A3SIM = bool(os.environ.get("PTOAS_ROOT"))
_skip_no_a2a3sim = pytest.mark.skipif(
    not _HAS_A2A3SIM,
    reason="a2a3sim not available (PTOAS_ROOT not set)",
)


class TestPhaseBMultiRegionEndToEnd:
    """2-region (static + dynamic) schedule round-trips and executes."""

    def _make_2_region_schedule(self) -> SonataScheduleContract:
        """Build a contract with 1 static + 1 dynamic region."""
        t1 = ScheduledTask(task_id=0, kernel_identity="add", func_id=1, core_type="aic",
            args=(ArgBinding(arg_identity="x", direction=ArgDirection.INPUT),
                  ArgBinding(arg_identity="y", direction=ArgDirection.OUTPUT)))
        t2 = ScheduledTask(task_id=1, kernel_identity="mul", func_id=2, core_type="aic",
            args=(ArgBinding(arg_identity="z", direction=ArgDirection.INPUT),))
        static_region = ScheduledRegion(
            region_id="r0", kind="static",
            tasks=(t1, t2),
            deps=(ScheduleDep(producer=0, consumer=1),),
        )
        dynamic_region = ScheduledRegion(
            region_id="r1", kind="dynamic",
            dynamic_mode="backend_dynamic",
        )
        return SonataScheduleContract(
            fingerprint="fp_multi_region",
            regions=(static_region, dynamic_region),
        )

    def test_binary_round_trip(self):
        """2-region schedule round-trips correctly through binary."""
        sched = self._make_2_region_schedule()
        data = sched.to_binary()
        assert len(data) > 0, "Binary is empty"

        sched2 = SonataScheduleContract.from_binary(data)
        assert sched2.fingerprint == "fp_multi_region"
        assert len(sched2.regions) == 2
        assert sched2.regions[0].kind == "static"
        assert sched2.regions[1].kind == "dynamic"
        assert len(sched2.regions[0].tasks) == 2
        assert len(sched2.regions[0].deps) == 1

    def test_json_round_trip(self):
        """2-region schedule round-trips correctly through JSON."""
        sched = self._make_2_region_schedule()
        d = sched.to_dict()
        assert len(d["regions"]) == 2

        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        sched2 = SonataScheduleContract.from_dict(d2)
        assert len(sched2.regions) == 2
        assert sched2.regions[0].kind == "static"
        assert sched2.regions[1].kind == "dynamic"

    def test_binary_deterministic(self):
        """Same 2-region schedule produces identical binary."""
        sched = self._make_2_region_schedule()
        b1 = sched.to_binary()
        b2 = sched.to_binary()
        assert b1 == b2

    @_skip_no_a2a3sim
    def test_static_st_test_passes_with_sonata(self):
        """Run a static multi-task ST test with --with-sonata."""
        test_file = PROJ_ROOT / "upstream" / "pypto" / "tests" / "st" / "runtime" / "scheduling" / "test_manual_scope_pipeline.py"
        if not test_file.exists():
            pytest.skip("ST test file not found")

        cmd = [
            sys.executable,
            "-m", "pytest",
            str(test_file),
            "--with-sonata", "--platform=a2a3sim",
            "--forked", "--no-header", "-q",
            "--rootdir", str(PROJ_ROOT),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"upstream/pypto/tests/st:{PROJ_ROOT / 'src'}"

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env, cwd=str(PROJ_ROOT))
        if result.returncode != 0:
            pytest.fail(f"Multi-region test failed (rc={result.returncode}):\n{result.stdout[:500]}")
