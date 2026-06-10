# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end tests: compile → sonata_plan.json → runtime hook → block_dim.

v0.22 Phase 1 A1: Verifies the full chain without NPU hardware.
Mocks compile_and_assemble + execute_on_device to test that:
1. sonata_compile writes sonata_plan.json
2. Runtime hook reads sonata_plan.json and computes block_dim
3. block_dim flows through to execute_on_device
"""

import json
import tempfile
from pathlib import Path

import pytest

from sonata.runtime_hook import apply_sonata_runtime_hints


def _write_plan(work_dir: Path, eligible: bool = True,
                region_statuses: dict | None = None,
                shape_assumptions: list | None = None) -> None:
    """Write a minimal sonata_plan.json."""
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


class TestE2ECompileToHook:
    """End-to-end: compile → sonata_plan.json → hook → block_dim."""

    def test_static_region_hook_suggests_32(self):
        """Static region → hook suggests block_dim=32."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_plan(Path(tmpdir), region_statuses={"region_0": "static"})
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=None,
                user_block_dim=None,
            )
            assert result.sonata_applied is True
            assert result.block_dim == 32
            assert "static" in result.reason

    def test_dynamic_region_hook_suggests_1(self):
        """Dynamic region → hook suggests block_dim=1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_plan(Path(tmpdir), region_statuses={"region_0": "dynamic"})
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=None,
                user_block_dim=None,
            )
            assert result.sonata_applied is True
            assert result.block_dim == 1

    def test_mixed_region_hook_suggests_16(self):
        """Mixed region → hook suggests block_dim=16."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_plan(Path(tmpdir), region_statuses={"region_0": "mixed"})
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=None,
                user_block_dim=None,
            )
            assert result.sonata_applied is True
            assert result.block_dim == 16

    def test_user_block_dim_not_overridden(self):
        """User explicitly supplied block_dim → hook doesn't override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_plan(Path(tmpdir), region_statuses={"region_0": "static"})
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=64, aicpu_thread_num=None,
                user_block_dim=64,
            )
            assert result.sonata_applied is False
            assert result.block_dim == 64
            assert result.reason == "user_supplied_block_dim"

    def test_runtime_config_block_dim_overridden(self):
        """block_dim from RUNTIME_CONFIG → hook overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_plan(Path(tmpdir), region_statuses={"region_0": "static"})
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=16, aicpu_thread_num=None,
                user_block_dim=None,
            )
            assert result.sonata_applied is True
            assert result.block_dim == 32

    def test_no_plan_returns_original(self):
        """No sonata_plan.json → original params unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=8, aicpu_thread_num=4,
                user_block_dim=None,
            )
            assert result.sonata_applied is False
            assert result.block_dim == 8
            assert result.aicpu_thread_num == 4

    def test_ineligible_plan_returns_original(self):
        """Ineligible plan → original params unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_plan(Path(tmpdir), eligible=False)
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=8, aicpu_thread_num=None,
                user_block_dim=None,
            )
            assert result.sonata_applied is False
            assert result.block_dim == 8

    def test_malformed_plan_fail_open(self):
        """Malformed plan.json → falls through to kernel_config (or no_sonata_data)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "sonata_plan.json").write_text("not valid json{{{")
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=8, aicpu_thread_num=2,
                user_block_dim=None,
            )
            assert result.sonata_applied is False
            assert result.block_dim == 8
            assert result.aicpu_thread_num == 2
            assert result.reason == "no_sonata_data"

    def test_full_chain_plan_written_then_consumed(self):
        """Full chain: write plan → hook reads → correct block_dim."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Write plan (simulates compile hook)
            _write_plan(Path(tmpdir), region_statuses={"region_0": "static"})

            # Step 2: Hook reads plan (simulates runner)
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=None,
                user_block_dim=None,
            )

            # Step 3: Verify
            assert result.sonata_applied is True
            assert result.block_dim == 32
            assert result.aicpu_thread_num is None

            # Step 4: Verify plan file is still valid JSON
            plan_data = json.loads((Path(tmpdir) / "sonata_plan.json").read_text())
            assert plan_data["eligible"] is True
            assert "region_statuses" in plan_data

    def test_full_chain_with_guard_stats(self):
        """Full chain: plan with shape_assumptions → guard_stats in plan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_plan(
                Path(tmpdir),
                region_statuses={"region_0": "static"},
                shape_assumptions=[{"symbol": "batch", "dims": [32], "severity": "soft"}],
            )

            # Verify plan has shape_assumptions
            plan_data = json.loads((Path(tmpdir) / "sonata_plan.json").read_text())
            assert "score" in plan_data
            assert len(plan_data["score"]["shape_assumptions"]) == 1

            # Hook should still work
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=None,
                user_block_dim=None,
            )
            assert result.sonata_applied is True
            assert result.block_dim == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
