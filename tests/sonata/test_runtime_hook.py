# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for Sonata runtime hook — the formal pre-dispatch integration point."""

import json
import tempfile
from pathlib import Path

import pytest

from sonata.runtime_hook import SonataRuntimeHints, apply_sonata_runtime_hints


class TestApplySonataRuntimeHints:
    """Core hook behavior tests."""

    def test_no_plan_returns_original(self):
        """No sonata data → original params unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=4,
            )
            assert result.sonata_applied is False
            assert result.reason == "no_sonata_data"
            assert result.block_dim is None
            assert result.aicpu_thread_num == 4

    def test_ineligible_plan_returns_original(self):
        """Ineligible plan → original params unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "sonata_plan.json"
            plan_path.write_text(json.dumps({"eligible": False}))
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=None,
            )
            assert result.sonata_applied is False
            assert result.reason == "plan_not_eligible"

    def test_user_supplied_block_dim_not_overridden(self):
        """User explicitly passed block_dim via user_block_dim → Sonata doesn't override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "sonata_plan.json"
            plan_path.write_text(json.dumps({
                "eligible": True,
                "region_statuses": {"region_0": "static"},
            }))
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=64, aicpu_thread_num=None,
                user_block_dim=64,
            )
            assert result.sonata_applied is False
            assert result.block_dim == 64
            assert result.reason == "user_supplied_block_dim"

    def test_runtime_config_block_dim_allows_override(self):
        """block_dim from RUNTIME_CONFIG (user_block_dim=None) → Sonata can override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "sonata_plan.json"
            plan_path.write_text(json.dumps({
                "eligible": True,
                "region_statuses": {"region_0": "static"},
            }))
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=16, aicpu_thread_num=None,
                user_block_dim=None,  # from RUNTIME_CONFIG, not user
            )
            assert result.sonata_applied is True
            assert result.block_dim == 32

    def test_static_region_suggests_block_dim(self):
        """Eligible plan with static region → Sonata suggests block_dim."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "sonata_plan.json"
            plan_path.write_text(json.dumps({
                "eligible": True,
                "region_statuses": {"region_0": "static"},
            }))
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=4,
            )
            assert result.sonata_applied is True
            assert result.block_dim == 32
            assert "static" in result.reason

    def test_dynamic_region_suggests_fallback(self):
        """Dynamic region → fallback block_dim=1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "sonata_plan.json"
            plan_path.write_text(json.dumps({
                "eligible": True,
                "region_statuses": {"region_0": "dynamic"},
            }))
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=None,
            )
            assert result.sonata_applied is True
            assert result.block_dim == 1
            assert "dynamic" in result.reason

    def test_mixed_region_conservative(self):
        """Mixed region → conservative block_dim=16."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "sonata_plan.json"
            plan_path.write_text(json.dumps({
                "eligible": True,
                "region_statuses": {"region_0": "mixed"},
            }))
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=None,
            )
            assert result.sonata_applied is True
            assert result.block_dim == 16
            assert "mixed" in result.reason

    def test_no_regions_returns_original(self):
        """Eligible plan with no regions → original params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "sonata_plan.json"
            plan_path.write_text(json.dumps({"eligible": True}))
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=None,
            )
            assert result.sonata_applied is False
            assert result.reason == "no_regions"

    def test_malformed_json_fail_open(self):
        """Malformed plan.json → falls through to kernel_config (or no_sonata_data)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "sonata_plan.json"
            plan_path.write_text("not valid json{{{")
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=8, aicpu_thread_num=2,
            )
            assert result.sonata_applied is False
            assert result.block_dim == 8
            assert result.aicpu_thread_num == 2
            # Corrupted plan falls through; no kernel_config → no_sonata_data
            assert result.reason == "no_sonata_data"

    def test_preserves_aicpu_thread_num(self):
        """aicpu_thread_num is preserved through the hook."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "sonata_plan.json"
            plan_path.write_text(json.dumps({
                "eligible": True,
                "region_statuses": {"region_0": "static"},
            }))
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=8,
            )
            assert result.aicpu_thread_num == 8


class TestKernelConfigFallback:
    """Tests for the kernel_config.py RUNTIME_CONFIG["sonata"] fallback (v0.22 Phase 2 B1)."""

    def _write_kernel_config(self, tmpdir: str, sonata_dict: dict | None = None) -> Path:
        """Helper: write a kernel_config.py with optional sonata key."""
        config_path = Path(tmpdir) / "kernel_config.py"
        lines = ['RUNTIME_CONFIG = {\n']
        lines.append('\t"runtime": "tensormap_and_ringbuffer",\n')
        lines.append('\t"block_dim": 16,\n')
        if sonata_dict is not None:
            sonata_repr = repr(sonata_dict)
            lines.append(f'\t"sonata": {sonata_repr},\n')
        lines.append('}\n')
        config_path.write_text(''.join(lines))
        return config_path

    def _write_plan(self, tmpdir: str, eligible: bool = True,
                    region_statuses: dict | None = None) -> Path:
        """Helper: write a sonata_plan.json."""
        plan_path = Path(tmpdir) / "sonata_plan.json"
        data: dict = {"eligible": eligible}
        if region_statuses is not None:
            data["region_statuses"] = region_statuses
        plan_path.write_text(json.dumps(data))
        return plan_path

    def test_kernel_config_eligible_uses_suggestions(self):
        """kernel_config with eligible config → uses suggested_block_dim."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_kernel_config(tmpdir, {
                "schema_version": 1, "eligible": True, "task_count": 5,
                "suggested_block_dim": 24,
            })
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=16, aicpu_thread_num=None,
            )
            assert result.sonata_applied is True
            assert result.block_dim == 24
            assert result.reason == "runtime_config_sonata"

    def test_kernel_config_ineligible(self):
        """kernel_config with eligible=False → not applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_kernel_config(tmpdir, {
                "schema_version": 1, "eligible": False, "task_count": 0,
            })
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=16, aicpu_thread_num=None,
            )
            assert result.sonata_applied is False
            assert result.reason == "plan_not_eligible"

    def test_kernel_config_missing_sonata_key(self):
        """kernel_config without sonata key → invalid config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_kernel_config(tmpdir, sonata_dict=None)
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=16, aicpu_thread_num=None,
            )
            assert result.sonata_applied is False
            assert result.reason == "invalid_sonata_config"

    def test_kernel_config_unknown_schema_version(self):
        """kernel_config with unknown schema_version → invalid config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_kernel_config(tmpdir, {
                "schema_version": 999, "eligible": True, "task_count": 5,
            })
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=16, aicpu_thread_num=None,
            )
            assert result.sonata_applied is False
            assert result.reason == "invalid_sonata_config"

    def test_kernel_config_no_suggestions(self):
        """kernel_config with no suggestions → not applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_kernel_config(tmpdir, {
                "schema_version": 1, "eligible": True, "task_count": 5,
            })
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=32, aicpu_thread_num=None,
            )
            assert result.sonata_applied is False
            assert result.reason == "no_suggestions"

    def test_kernel_config_user_block_dim_precedence(self):
        """user_block_dim takes precedence over kernel_config suggestion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_kernel_config(tmpdir, {
                "schema_version": 1, "eligible": True, "task_count": 5,
                "suggested_block_dim": 24,
            })
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=64, aicpu_thread_num=None,
                user_block_dim=64,
            )
            assert result.sonata_applied is False
            assert result.reason == "user_supplied_block_dim"

    def test_kernel_config_no_plan_json_fallback(self):
        """Only kernel_config exists (no sonata_plan.json) → fallback works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_kernel_config(tmpdir, {
                "schema_version": 1, "eligible": True, "task_count": 3,
                "suggested_block_dim": 32,
            })
            # No sonata_plan.json written
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=16, aicpu_thread_num=4,
            )
            assert result.sonata_applied is True
            assert result.block_dim == 32
            assert result.aicpu_thread_num == 4

    def test_plan_json_takes_precedence(self):
        """sonata_plan.json takes precedence over kernel_config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_kernel_config(tmpdir, {
                "schema_version": 1, "eligible": True, "task_count": 5,
                "suggested_block_dim": 999,  # should NOT be used
            })
            self._write_plan(tmpdir, region_statuses={"region_0": "static"})
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=None,
            )
            assert result.sonata_applied is True
            assert result.block_dim == 32  # from plan dispatch, not 999

    def test_corrupted_plan_with_kernel_config_fallback(self):
        """Corrupted plan.json falls through to valid kernel_config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Corrupted plan.json
            (Path(tmpdir) / "sonata_plan.json").write_text("{{{bad json}}")
            # Valid kernel_config with fallback data
            self._write_kernel_config(tmpdir, {
                "schema_version": 1, "eligible": True, "task_count": 5,
                "suggested_block_dim": 32,
            })
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=16, aicpu_thread_num=None,
            )
            assert result.sonata_applied is True
            assert result.block_dim == 32
            assert result.reason == "runtime_config_sonata"

    def test_kernel_config_bad_file_fail_open(self):
        """Malformed kernel_config.py → fail open."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "kernel_config.py"
            config_path.write_text("this is not valid python {{")
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=8, aicpu_thread_num=2,
            )
            assert result.sonata_applied is False
            assert result.block_dim == 8

    def test_kernel_config_aicpu_suggestion(self):
        """kernel_config with aicpu_thread_num suggestion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_kernel_config(tmpdir, {
                "schema_version": 1, "eligible": True, "task_count": 5,
                "suggested_block_dim": 24,
                "suggested_aicpu_thread_num": 8,
            })
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=16, aicpu_thread_num=4,
            )
            assert result.sonata_applied is True
            assert result.block_dim == 24
            assert result.aicpu_thread_num == 8


class TestSonataRuntimeHints:
    """SonataRuntimeHints dataclass tests."""

    def test_frozen(self):
        hints = SonataRuntimeHints(block_dim=32, aicpu_thread_num=4,
                                   sonata_applied=True, reason="test")
        with pytest.raises(AttributeError):
            hints.block_dim = 64

    def test_fields(self):
        hints = SonataRuntimeHints(block_dim=None, aicpu_thread_num=None,
                                   sonata_applied=False, reason="no_plan")
        assert hints.block_dim is None
        assert hints.sonata_applied is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
