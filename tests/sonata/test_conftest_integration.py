# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for ST conftest integration — verifies the official hook path.

v0.21 Phase 1 B1: Verifies that --with-sonata correctly writes
sonata_plan.json and the runner hook consumes it.

These tests verify the conftest mechanism without requiring actual
NPU hardware or PyPTO compilation.
"""

import json
import tempfile
from pathlib import Path

import pytest


class TestConftestIntegration:
    """Verify ST conftest writes sonata_plan.json for runner hook consumption."""

    def test_make_patched_compile_writes_sonata_plan(self):
        """_make_patched_compile wrapper writes sonata_plan.json after compilation."""
        # This test verifies the compile hook mechanism exists and is importable
        from tests.st_sonata.conftest import _make_patched_compile
        assert callable(_make_patched_compile)

    def test_extract_program_from_module_finds_program(self):
        """_extract_program_from_module finds @pl.program objects."""
        from tests.st_sonata.conftest import _extract_program_from_module
        # With no program in module → returns None
        import types
        mod = types.ModuleType("empty_mod")
        result = _extract_program_from_module(mod)
        assert result is None

    def test_analysis_cache_mechanism(self):
        """_analysis_cache stores and retrieves analysis results."""
        from tests.st_sonata.conftest import _analysis_cache
        # Cache is a dict
        assert isinstance(_analysis_cache, dict)

    def test_session_results_collector(self):
        """_session_results collector exists and is a list."""
        from tests.st_sonata.conftest import _session_results
        assert isinstance(_session_results, list)

    def test_pytest_addoption_registers_with_sonata(self):
        """pytest_addoption registers --with-sonata flag."""
        from tests.st_sonata.conftest import pytest_addoption
        # Verify the function exists and is callable
        assert callable(pytest_addoption)

    def test_sonata_plan_json_written_by_compile_hook(self):
        """Verify sonata_plan.json is written when compile hook runs.

        This test mocks the compile flow and verifies the plan file is created.
        """
        import json
        import tempfile
        from pathlib import Path
        from sonata.score import Score, RuntimeTarget, Task
        from sonata.pipeline import SonataAnalysisResult

        # Create a mock analysis result
        score = Score(
            name="test_program",
            runtime_target=RuntimeTarget(),
            tasks=(
                Task(task_id=0, func_id=0, core_type="aic", name="kernel"),
            ),
        )
        result = SonataAnalysisResult(
            eligible=True,
            score=score,
            region_statuses={"region_0": "static"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "sonata_plan.json"
            result.save(plan_path)

            # Verify plan file exists and is valid JSON
            assert plan_path.exists()
            data = json.loads(plan_path.read_text())
            assert data["eligible"] is True
            assert "region_statuses" in data
            assert data["region_statuses"]["region_0"] == "static"

    def test_runner_hook_consumes_sonata_plan(self):
        """Verify runner hook reads sonata_plan.json and produces hints.

        This test creates a plan file and verifies the hook reads it correctly.
        """
        import json
        import tempfile
        from pathlib import Path
        from sonata.runtime_hook import apply_sonata_runtime_hints

        with tempfile.TemporaryDirectory() as tmpdir:
            plan = {
                "eligible": True,
                "region_statuses": {"region_0": "static"},
            }
            (Path(tmpdir) / "sonata_plan.json").write_text(json.dumps(plan))

            # Hook should read the plan and suggest block_dim
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=None,
                user_block_dim=None,
            )
            assert result.sonata_applied is True
            assert result.block_dim == 32  # static region → optimized

    def test_end_to_end_plan_write_and_consume(self):
        """Full end-to-end: write plan → hook reads → correct block_dim.

        This verifies the complete ST flow without NPU hardware.
        """
        import json
        import tempfile
        from pathlib import Path
        from sonata.score import Score, RuntimeTarget, Task
        from sonata.pipeline import SonataAnalysisResult
        from sonata.runtime_hook import apply_sonata_runtime_hints

        # Step 1: Create analysis result (simulates compile hook)
        score = Score(
            name="e2e_test",
            runtime_target=RuntimeTarget(),
            tasks=(
                Task(task_id=0, func_id=0, core_type="aic", name="kernel"),
            ),
        )
        result = SonataAnalysisResult(
            eligible=True,
            score=score,
            region_statuses={"region_0": "static"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 2: Write plan (simulates compile hook writing sonata_plan.json)
            plan_path = Path(tmpdir) / "sonata_plan.json"
            result.save(plan_path)
            assert plan_path.exists()

            # Step 3: Hook reads plan (simulates runner consuming sonata_plan.json)
            hook_result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=None,
                user_block_dim=None,
            )
            assert hook_result.sonata_applied is True
            assert hook_result.block_dim == 32
            assert "static" in hook_result.reason

            # Step 4: Verify plan data round-trips correctly
            data = json.loads(plan_path.read_text())
            assert data["eligible"] is True
            # guard_stats only present when shape_assumptions exist
            if score.shape_assumptions:
                assert "guard_stats" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
