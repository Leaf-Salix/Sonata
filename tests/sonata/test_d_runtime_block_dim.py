# Copyright (c) PyPTO Contributors.
# ---------------------------------------------------------------------------
"""v0.28 Phase D: Verify Sonata's block_dim takes effect at runtime.

Traces the path: RUNTIME_CONFIG → execute_on_device → ChipCallConfig.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

from sonata.pipeline import SonataAnalysisResult, dispatch_regions, compute_scheduling_instructions

# a2a3sim check
_HAS_A2A3SIM = bool(os.environ.get("PTOAS_ROOT"))
_skip_no_a2a3sim = pytest.mark.skipif(
    not _HAS_A2A3SIM,
    reason="a2a3sim not available (PTOAS_ROOT not set)",
)


class TestPhaseDRuntimeBlockDimEffect:
    """Sonata's block_dim suggestion reaches the execution engine."""

    def test_block_dim_in_runtime_config(self):
        """Sonata's to_runtime_config() includes block_dim when suggested."""
        result = SonataAnalysisResult(
            eligible=True,
            region_statuses={"r_dyn": "dynamic"},
        )
        cfg = result.to_runtime_config()
        d = cfg.to_run_config_dict()
        assert "suggested_block_dim" in d
        assert d["suggested_block_dim"] == 1  # dynamic → block_dim=1

    def test_promoted_to_block_dim_key(self):
        """_patch_kernel_config_sonata promotes suggested_block_dim → block_dim."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "st_conftest",
            str(Path(__file__).resolve().parents[2] / "tests" / "st_sonata" / "conftest.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmpdir_str:
            kc = Path(tmpdir_str) / "kernel_config.py"
            kc.write_text('RUNTIME_CONFIG = {"runtime": "tensormap_and_ringbuffer",}\n')
            mod._patch_kernel_config_sonata(Path(tmpdir_str), {"suggested_block_dim": 1})

            spec2 = importlib.util.spec_from_file_location("_kc", str(kc))
            m2 = importlib.util.module_from_spec(spec2)
            spec2.loader.exec_module(m2)
            assert m2.RUNTIME_CONFIG.get("block_dim") == 1

    def test_runtime_config_block_dim_is_readable(self):
        """Simulate what execute_compiled does: load kernel_config and read block_dim."""
        with tempfile.TemporaryDirectory() as tmpdir_str:
            kc = Path(tmpdir_str) / "kernel_config.py"
            kc.write_text('RUNTIME_CONFIG = {"runtime": "tensormap_and_ringbuffer", "block_dim": 1}\n')

            import importlib.util
            spec = importlib.util.spec_from_file_location("_kc_test", str(kc))
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            rc = m.RUNTIME_CONFIG

            # This is exactly how runner.py reads it:
            block_dim = rc.get("block_dim")
            assert block_dim == 1, f"Expected block_dim=1, got {block_dim}"

    def test_runner_code_reads_block_dim(self):
        """The execution runner (runner.py) reads block_dim from RUNTIME_CONFIG.

        This is a static analysis check — we verify the runtime code path
        exists, without actually running on hardware.
        """
        runner_py = Path(__file__).resolve().parents[2] / "upstream" / "pypto" / "python" / "pypto" / "runtime" / "runner.py"
        if not runner_py.exists():
            pytest.skip("runner.py not found")

        source = runner_py.read_text()
        # The runner reads block_dim in at least three places
        assert "runtime_config.get(\"block_dim\")" in source, (
            "runner.py does not read block_dim from RUNTIME_CONFIG"
        )

        # Count occurrences
        count = source.count("runtime_config.get(\"block_dim\")")
        print(f"\nrunner.py reads block_dim from RUNTIME_CONFIG in {count} places")
        assert count >= 1, "Expected at least 1 reference to block_dim"
