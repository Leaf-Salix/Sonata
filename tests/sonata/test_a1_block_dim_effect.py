# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""v0.28 Phase A1: Prove Sonata changes block_dim from default.

Creates a minimal mixed-region scenario (dynamic region first), verifies
Sonata suggests block_dim=1 (not default 32), and confirms the injection
path works end-to-end.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from sonata.pipeline import SonataAnalysisResult, dispatch_regions, compute_scheduling_instructions

PROJ_ROOT = Path(__file__).resolve().parents[2]

# a2a3sim check
_HAS_A2A3SIM = bool(os.environ.get("PTOAS_ROOT"))
_skip_no_a2a3sim = pytest.mark.skipif(
    not _HAS_A2A3SIM,
    reason="a2a3sim not available (PTOAS_ROOT not set)",
)


class TestPhaseA1BlockDimEffect:
    """Prove Sonata's block_dim suggestion differs from default for dynamic regions."""

    def test_dynamic_region_suggests_block_dim_1(self):
        """Dynamic region → Sonata suggests block_dim=1 (not default 32)."""
        result = SonataAnalysisResult(
            eligible=True,
            region_statuses={"r_dyn": "dynamic"},
        )
        instructions = compute_scheduling_instructions(dispatch_regions(result))
        assert instructions[0].block_dim == 1, (
            f"Expected block_dim=1 for dynamic region, got {instructions[0].block_dim}"
        )
        assert "fallback" in instructions[0].reason.lower()

    def test_static_region_suggests_block_dim_32(self):
        """Static region → Sonata suggests block_dim=32 (matches default)."""
        result = SonataAnalysisResult(
            eligible=True,
            region_statuses={"r_static": "static"},
        )
        instructions = compute_scheduling_instructions(dispatch_regions(result))
        assert instructions[0].block_dim == 32

    def test_mixed_first_dynamic_suggests_block_dim_1(self):
        """If first region is dynamic, suggested_block_dim=1 (not 32)."""
        result = SonataAnalysisResult(
            eligible=True,
            region_statuses={"r0_dyn": "dynamic", "r1_static": "static"},
        )
        d = result.to_runtime_config().to_run_config_dict()
        assert d.get("suggested_block_dim") == 1

    def test_mixed_first_static_suggests_block_dim_32(self):
        """If first region is static, suggested_block_dim=32 (matches default)."""
        result = SonataAnalysisResult(
            eligible=True,
            region_statuses={"r0_static": "static", "r1_dyn": "dynamic"},
        )
        d = result.to_runtime_config().to_run_config_dict()
        assert d.get("suggested_block_dim") == 32

    def test_kernel_config_injection_promotes_block_dim(self):
        """_patch_kernel_config_sonata promotes suggested_block_dim to block_dim."""
        spec = importlib.util.spec_from_file_location(
            "st_conftest",
            str(PROJ_ROOT / "tests" / "st_sonata" / "conftest.py"),
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
            assert m2.RUNTIME_CONFIG.get("block_dim") == 1, (
                f"Expected block_dim=1 after injection, got {m2.RUNTIME_CONFIG.get('block_dim')}"
            )

    @_skip_no_a2a3sim
    def test_dynamic_test_runs_with_sonata_no_failures(self):
        """Run a dynamic ST test with --with-sonata; verify no failures."""
        test_file = PROJ_ROOT / "upstream" / "pypto" / "tests" / "st" / "runtime" / "control_flow" / "test_dynamic_shape.py"
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
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"upstream/pypto/tests/st:{PROJ_ROOT / 'src'}"
        if existing_pythonpath:
            env["PYTHONPATH"] += f":{existing_pythonpath}"

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env, cwd=str(PROJ_ROOT))
        if result.returncode != 0:
            pytest.fail(f"Dynamic test with --with-sonata failed (rc={result.returncode}):\n{result.stdout[:500]}")
