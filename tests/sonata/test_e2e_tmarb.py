"""End-to-end test: sonata_compile() produces tmarb_call_trace.json alongside artifacts."""
import json
import tempfile
from pathlib import Path

import shutil

import pytest

from sonata.schedule import (
    ArgBinding,
    ScheduledRegion,
    ScheduledTask,
    SonataScheduleContract,
)


class TestE2ETMARBTrace:
    def test_trace_content_static_region(self):
        """Verify trace content for a simple static region."""
        t1 = ScheduledTask(task_id=0, kernel_identity="add", func_id=3, core_type="aic",
            args=(ArgBinding(arg_identity="x"),), outputs=("z",))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,))
        schedule = SonataScheduleContract(fingerprint="fp", regions=(r0,))

        from sonata.mapping.trace import generate_trace, trace_to_json
        trace = generate_trace(schedule)
        trace_json = trace_to_json(trace)
        data = json.loads(trace_json)

        assert len(data) >= 3  # PTO2_SCOPE + add_input + submit
        apis = [e["api"] for e in data]
        assert "PTO2_SCOPE" in apis
        assert "rt_submit_aic_task" in apis

    def test_trace_content_dynamic_region(self):
        """Verify trace content for a dynamic region."""
        r0 = ScheduledRegion(region_id="r0", kind="dynamic", dynamic_mode="backend_dynamic")
        schedule = SonataScheduleContract(fingerprint="fp", regions=(r0,))

        from sonata.mapping.trace import generate_trace, trace_to_json
        trace = generate_trace(schedule)
        data = json.loads(trace_to_json(trace))
        assert len(data) == 1
        assert data[0]["api"] == "PTO2_SCOPE"
        assert data[0]["region_kind"] == "dynamic"

    def test_trace_deterministic(self):
        """Same schedule → same trace JSON."""
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic",
            args=(ArgBinding(arg_identity="x"),))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,))
        schedule = SonataScheduleContract(fingerprint="fp", regions=(r0,))

        from sonata.mapping.trace import generate_trace, trace_to_json
        j1 = trace_to_json(generate_trace(schedule))
        j2 = trace_to_json(generate_trace(schedule))
        assert j1 == j2


# -- Gated tests (require pypto) --

try:
    import pypto  # noqa: F401
    _HAS_PYPTO = True
except ImportError:
    _HAS_PYPTO = False

_HAS_PTOAS = shutil.which("ptoas") is not None


@pytest.mark.skipif(not _HAS_PYPTO or not _HAS_PTOAS, reason="pypto or ptoas not available")
class TestE2EFullPipeline:
    """Full sonata_compile() → tmarb_call_trace.json pipeline.

    Requires pypto to be installed.
    """

    def test_sonata_compile_produces_trace(self):
        """Verify sonata_compile() writes tmarb_call_trace.json."""
        import tempfile
        from pathlib import Path
        import pypto.language as pl
        import pypto.ir as pir

        M, N = 16, 16

        @pl.program
        class SimpleProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, a: pl.Tensor[[M, N], pl.FP32],
                       out: pl.Out[pl.Tensor[[M, N], pl.FP32]]):
                tile_a = pl.load(a, [0, 0], [M, N])
                tile_c = pl.tile.abs(tile_a)
                pl.store(tile_c, [0, 0], out)
                return out
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[M, N], pl.FP32],
                             out: pl.Out[pl.Tensor[[M, N], pl.FP32]]):
                self.kernel(a, out)

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            compiled = pir.compile(SimpleProgram, output_dir=str(work_dir))

            from sonata.pipeline import sonata_analyze, _extract_certified_ir
            certified_ir = _extract_certified_ir(SimpleProgram)
            result = sonata_analyze(certified_ir)

            # Call the internal writer
            from sonata.pipeline import _write_bound_schedule
            _write_bound_schedule(result, compiled, work_dir)

            # Verify trace was written
            trace_path = work_dir / "tmarb_call_trace.json"
            assert trace_path.exists(), f"trace not found at {trace_path}"
            trace_data = json.loads(trace_path.read_text())
            assert isinstance(trace_data, list)
            assert len(trace_data) > 0

            # Verify the entries make sense
            apis = {e["api"] for e in trace_data}
            assert "PTO2_SCOPE" in apis
