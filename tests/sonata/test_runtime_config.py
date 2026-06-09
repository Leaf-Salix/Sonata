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


class TestToRuntimeConfig:
    """v0.22 Phase 1 A2: SonataAnalysisResult.to_runtime_config() tests."""

    def test_eligible_result_produces_config(self):
        """Eligible result with score produces valid SonataRuntimeConfig."""
        import warnings
        from sonata.score import Score, RuntimeTarget, Task, ShapeAssumption
        from sonata.pipeline import SonataAnalysisResult

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            score = Score(
                name="test",
                runtime_target=RuntimeTarget(),
                tasks=(
                    Task(task_id=0, func_id=0, core_type="aic", name="kernel"),
                    Task(task_id=1, func_id=1, core_type="aiv", name="add"),
                ),
                shape_assumptions=(
                    ShapeAssumption(symbol="batch", dims=(32,)),
                ),
            )
        result = SonataAnalysisResult(
            eligible=True,
            score=score,
            region_statuses={"region_0": "static"},
        )
        cfg = result.to_runtime_config()
        assert cfg.eligible is True
        assert cfg.task_count == 2
        assert cfg.suggested_block_dim == 32  # static region
        assert cfg.guard_count == 1
        assert cfg.guard_symbols == ("batch",)
        assert cfg.region_statuses == {"region_0": "static"}

    def test_ineligible_result_produces_config(self):
        """Ineligible result produces config with eligible=False."""
        from sonata.pipeline import SonataAnalysisResult

        result = SonataAnalysisResult(eligible=False)
        cfg = result.to_runtime_config()
        assert cfg.eligible is False
        assert cfg.task_count == 0
        assert cfg.suggested_block_dim is None

    def test_config_includes_memory_peak(self):
        """Config includes memory_peak_bytes from memory_plan."""
        from sonata.pipeline import SonataAnalysisResult
        from sonata.memory_plan import MemoryPlan

        mp = MemoryPlan(allocations=(), peak_memory=2048)
        result = SonataAnalysisResult(eligible=True, memory_plan=mp)
        cfg = result.to_runtime_config()
        assert cfg.memory_peak_bytes == 2048

    def test_config_no_memory_plan(self):
        """Config with no memory_plan → memory_peak_bytes is None."""
        from sonata.pipeline import SonataAnalysisResult

        result = SonataAnalysisResult(eligible=True)
        cfg = result.to_runtime_config()
        assert cfg.memory_peak_bytes is None

    def test_config_to_dict_roundtrip(self):
        """Config to_dict produces valid JSON-serializable dict."""
        from sonata.runtime_config import SonataRuntimeConfig

        cfg = SonataRuntimeConfig(
            eligible=True, task_count=5,
            suggested_block_dim=24, guard_count=2,
            guard_symbols=("batch", "seq"),
        )
        d = cfg.to_run_config_dict()
        assert d["eligible"] is True
        assert d["suggested_block_dim"] == 24
        assert isinstance(d["guard_symbols"], list)


class TestPatchKernelConfig:
    """v0.22 Phase 1 A3: _patch_kernel_config_sonata() tests."""

    def test_injects_sonata_key(self):
        """Injects sonata dict into RUNTIME_CONFIG."""
        from tests.st_sonata.conftest import _patch_kernel_config_sonata
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "kernel_config.py"
            config_path.write_text(
                'RUNTIME_CONFIG = {\n'
                '\t"runtime": "tensormap_and_ringbuffer",\n'
                '\t"block_dim": 32,\n'
                '}\n'
            )
            sonata_dict = {"eligible": True, "task_count": 5}
            _patch_kernel_config_sonata(Path(tmpdir), sonata_dict)

            result = config_path.read_text()
            assert '"sonata"' in result
            assert "eligible" in result
            assert "task_count" in result
            # Check Python-native types (True, not true)
            assert "True" in result
            assert "true" not in result

    def test_no_config_file_does_not_crash(self):
        """Missing kernel_config.py → no crash, just returns."""
        from tests.st_sonata.conftest import _patch_kernel_config_sonata
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # No kernel_config.py
            _patch_kernel_config_sonata(Path(tmpdir), {"eligible": True})
            # Should not raise

    def test_no_runtime_config_does_not_crash(self):
        """kernel_config.py without RUNTIME_CONFIG → no crash."""
        from tests.st_sonata.conftest import _patch_kernel_config_sonata
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "kernel_config.py"
            config_path.write_text("# no runtime config here\n")
            _patch_kernel_config_sonata(Path(tmpdir), {"eligible": True})
            # Should not raise

    def test_preserves_existing_config(self):
        """Existing RUNTIME_CONFIG keys are preserved."""
        from tests.st_sonata.conftest import _patch_kernel_config_sonata
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "kernel_config.py"
            config_path.write_text(
                'RUNTIME_CONFIG = {\n'
                '\t"runtime": "tensormap_and_ringbuffer",\n'
                '\t"block_dim": 32,\n'
                '}\n'
            )
            _patch_kernel_config_sonata(Path(tmpdir), {"eligible": True})

            result = config_path.read_text()
            assert '"runtime"' in result
            assert '"block_dim"' in result
            assert '"sonata"' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
