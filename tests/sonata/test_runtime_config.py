# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for Sonata RUNTIME_CONFIG schema (v0.22 Phase 1 A1)."""

import pytest

from sonata.runtime_config import SONATA_RUNTIME_CONFIG_SCHEMA_VERSION, SonataRuntimeConfig


class TestSonataRuntimeConfig:
    """SonataRuntimeConfig dataclass tests."""

    def test_minimal_config(self):
        """Minimal config emits required fields only."""
        cfg = SonataRuntimeConfig(eligible=True, task_count=5)
        d = cfg.to_run_config_dict()
        assert d["schema_version"] == SONATA_RUNTIME_CONFIG_SCHEMA_VERSION
        assert d["eligible"] is True
        assert d["task_count"] == 5
        # Optional fields not present when None/empty
        assert "suggested_block_dim" not in d
        assert "suggested_aicpu_thread_num" not in d
        assert "memory_peak_bytes" not in d
        assert "region_statuses" not in d
        assert "guard_count" not in d

    def test_full_config(self):
        """Full config emits all fields."""
        cfg = SonataRuntimeConfig(
            eligible=True,
            task_count=49,
            suggested_block_dim=24,
            suggested_aicpu_thread_num=4,
            memory_peak_bytes=1048576,
            region_statuses={"root": "static"},
            guard_count=3,
            guard_symbols=("batch_size", "seq_len", "hidden"),
        )
        d = cfg.to_run_config_dict()
        assert d["suggested_block_dim"] == 24
        assert d["suggested_aicpu_thread_num"] == 4
        assert d["memory_peak_bytes"] == 1048576
        assert d["region_statuses"] == {"root": "static"}
        assert d["guard_count"] == 3
        assert d["guard_symbols"] == ["batch_size", "seq_len", "hidden"]

    def test_frozen(self):
        """Config is frozen (immutable)."""
        cfg = SonataRuntimeConfig(eligible=True, task_count=1)
        with pytest.raises(AttributeError):
            cfg.task_count = 2

    def test_schema_version_constant(self):
        """Schema version constant is 1."""
        assert SONATA_RUNTIME_CONFIG_SCHEMA_VERSION == 1

    def test_default_schema_version(self):
        """Default schema_version matches constant."""
        cfg = SonataRuntimeConfig(eligible=True, task_count=0)
        assert cfg.schema_version == SONATA_RUNTIME_CONFIG_SCHEMA_VERSION

    def test_empty_region_statuses_omitted(self):
        """Empty region_statuses dict is omitted from output."""
        cfg = SonataRuntimeConfig(eligible=True, task_count=1, region_statuses={})
        d = cfg.to_run_config_dict()
        assert "region_statuses" not in d

    def test_guard_symbols_as_list_in_dict(self):
        """guard_symbols tuple emitted as list in dict (JSON-safe)."""
        cfg = SonataRuntimeConfig(
            eligible=True, task_count=1,
            guard_count=2, guard_symbols=("a", "b"),
        )
        d = cfg.to_run_config_dict()
        assert isinstance(d["guard_symbols"], list)
        assert d["guard_symbols"] == ["a", "b"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
