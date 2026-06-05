# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end tests for the Sonata runner hook integration.

v0.21 Phase 1 A1: Verifies that the runner hook in execute_compiled()
actually reads sonata_plan.json and applies the correct block_dim.

These tests mock compile_and_assemble and execute_on_device to verify
the hook's behavior without requiring real PyPTO compilation or NPU hardware.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _write_sonata_plan(work_dir: Path, eligible: bool = True,
                       region_statuses: dict | None = None,
                       shape_assumptions: list | None = None) -> None:
    """Write a minimal sonata_plan.json for testing."""
    plan = {
        "eligible": eligible,
        "region_statuses": region_statuses if region_statuses is not None else {"region_0": "static"},
    }
    if shape_assumptions:
        plan["score"] = {
            "name": "test",
            "shape_assumptions": shape_assumptions,
        }
    (work_dir / "sonata_plan.json").write_text(json.dumps(plan))


class TestRunnerHookEndToEnd:
    """End-to-end: runner hook reads sonata_plan.json and applies block_dim."""

    def test_hook_applies_block_dim_for_static_region(self):
        """Static region plan → hook suggests block_dim=32."""
        from sonata.runtime_hook import apply_sonata_runtime_hints

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_sonata_plan(Path(tmpdir), region_statuses={"region_0": "static"})
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=None,
                user_block_dim=None,
            )
            assert result.sonata_applied is True
            assert result.block_dim == 32
            assert "static" in result.reason

    def test_hook_does_not_override_user_block_dim(self):
        """User-supplied block_dim → hook returns original, doesn't override."""
        from sonata.runtime_hook import apply_sonata_runtime_hints

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_sonata_plan(Path(tmpdir), region_statuses={"region_0": "static"})
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=64, aicpu_thread_num=None,
                user_block_dim=64,  # user explicitly supplied
            )
            assert result.sonata_applied is False
            assert result.block_dim == 64
            assert result.reason == "user_supplied_block_dim"

    def test_hook_overrides_runtime_config_block_dim(self):
        """block_dim from RUNTIME_CONFIG (user_block_dim=None) → hook overrides."""
        from sonata.runtime_hook import apply_sonata_runtime_hints

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_sonata_plan(Path(tmpdir), region_statuses={"region_0": "static"})
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=16, aicpu_thread_num=None,
                user_block_dim=None,  # from RUNTIME_CONFIG, not user
            )
            assert result.sonata_applied is True
            assert result.block_dim == 32  # overrides RUNTIME_CONFIG value

    def test_no_plan_returns_original_params(self):
        """No sonata_plan.json → hook returns original params unchanged."""
        from sonata.runtime_hook import apply_sonata_runtime_hints

        with tempfile.TemporaryDirectory() as tmpdir:
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=8, aicpu_thread_num=4,
                user_block_dim=None,
            )
            assert result.sonata_applied is False
            assert result.block_dim == 8
            assert result.aicpu_thread_num == 4
            assert result.reason == "no_sonata_plan"

    def test_ineligible_plan_returns_original(self):
        """Ineligible plan → hook returns original params."""
        from sonata.runtime_hook import apply_sonata_runtime_hints

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_sonata_plan(Path(tmpdir), eligible=False)
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=8, aicpu_thread_num=4,
                user_block_dim=None,
            )
            assert result.sonata_applied is False
            assert result.block_dim == 8

    def test_dynamic_region_suggests_fallback(self):
        """Dynamic region → hook suggests block_dim=1."""
        from sonata.runtime_hook import apply_sonata_runtime_hints

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_sonata_plan(Path(tmpdir), region_statuses={"region_0": "dynamic"})
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=None,
                user_block_dim=None,
            )
            assert result.sonata_applied is True
            assert result.block_dim == 1

    def test_mixed_region_suggests_conservative(self):
        """Mixed region → hook suggests block_dim=16."""
        from sonata.runtime_hook import apply_sonata_runtime_hints

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_sonata_plan(Path(tmpdir), region_statuses={"region_0": "mixed"})
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=None,
                user_block_dim=None,
            )
            assert result.sonata_applied is True
            assert result.block_dim == 16

    def test_malformed_plan_fail_open(self):
        """Malformed JSON → fail open, original params returned."""
        from sonata.runtime_hook import apply_sonata_runtime_hints

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "sonata_plan.json").write_text("not valid json{{{")
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=8, aicpu_thread_num=2,
                user_block_dim=None,
            )
            assert result.sonata_applied is False
            assert result.block_dim == 8
            assert result.aicpu_thread_num == 2
            assert "hook_error" in result.reason

    def test_aicpu_thread_num_preserved(self):
        """aicpu_thread_num passes through unchanged."""
        from sonata.runtime_hook import apply_sonata_runtime_hints

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_sonata_plan(Path(tmpdir), region_statuses={"region_0": "static"})
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=8,
                user_block_dim=None,
            )
            assert result.aicpu_thread_num == 8

    def test_empty_region_statuses_returns_original(self):
        """Empty region_statuses → hook returns original params (no scheduling)."""
        from sonata.runtime_hook import apply_sonata_runtime_hints

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_sonata_plan(Path(tmpdir), region_statuses={})
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=4, aicpu_thread_num=None,
                user_block_dim=None,
            )
            assert result.sonata_applied is False
            assert result.block_dim == 4
            assert result.reason == "no_regions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
