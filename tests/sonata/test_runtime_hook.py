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
        """No sonata_plan.json → original params unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=4,
            )
            assert result.sonata_applied is False
            assert result.reason == "no_sonata_plan"
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
        """Malformed JSON → fail open, original params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "sonata_plan.json"
            plan_path.write_text("not valid json{{{")
            result = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=8, aicpu_thread_num=2,
            )
            assert result.sonata_applied is False
            assert result.block_dim == 8
            assert result.aicpu_thread_num == 2
            assert "hook_error" in result.reason

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
