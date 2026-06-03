# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for operator profiling (v0.18 Phase 3)."""

import tempfile
from pathlib import Path

import pytest

from sonata.profile import OperatorProfile, ProfileDatabase


class TestOperatorProfile:
    """OperatorProfile dataclass tests."""

    def test_construction(self):
        p = OperatorProfile(
            op_signature="matmul_fp16_128x128x128",
            op_type="matmul", shape=(128, 128, 128), dtype="fp16",
            core_type="aic", mean_latency_us=320.0, std_latency_us=12.5,
            sample_count=150,
        )
        assert p.op_type == "matmul"
        assert p.mean_latency_us == 320.0

    def test_frozen(self):
        p = OperatorProfile(
            op_signature="x", op_type="x", shape=(), dtype="fp16",
            core_type="aic", mean_latency_us=0.0, std_latency_us=0.0,
            sample_count=0,
        )
        with pytest.raises(AttributeError):
            p.mean_latency_us = 100.0


class TestProfileDatabase:
    """ProfileDatabase record/lookup/persistence tests."""

    def test_lookup_returns_none_for_unknown(self):
        db = ProfileDatabase()
        assert db.lookup("matmul", (128, 128), "fp16") is None

    def test_record_and_lookup(self):
        db = ProfileDatabase()
        db.record("matmul", (128, 128), "fp16", "aic", 320.0)
        profile = db.lookup("matmul", (128, 128), "fp16")
        assert profile is not None
        assert profile.mean_latency_us == 320.0
        assert profile.sample_count == 1

    def test_multiple_records_update_mean(self):
        db = ProfileDatabase()
        db.record("matmul", (128, 128), "fp16", "aic", 300.0)
        db.record("matmul", (128, 128), "fp16", "aic", 340.0)
        profile = db.lookup("matmul", (128, 128), "fp16")
        assert profile.sample_count == 2
        assert abs(profile.mean_latency_us - 320.0) < 0.01

    def test_different_shapes_are_separate(self):
        db = ProfileDatabase()
        db.record("matmul", (128, 128), "fp16", "aic", 320.0)
        db.record("matmul", (64, 64), "fp16", "aic", 80.0)
        assert db.lookup("matmul", (128, 128), "fp16").mean_latency_us == 320.0
        assert db.lookup("matmul", (64, 64), "fp16").mean_latency_us == 80.0

    def test_all_profiles(self):
        db = ProfileDatabase()
        db.record("matmul", (128, 128), "fp16", "aic", 320.0)
        db.record("conv2d", (56, 56), "fp16", "aic", 1050.0)
        all_p = db.all_profiles()
        assert len(all_p) == 2

    def test_save_and_load_round_trip(self):
        db = ProfileDatabase()
        db.record("matmul", (128, 128), "fp16", "aic", 320.0)
        db.record("conv2d", (56, 56), "fp16", "aic", 1050.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.json"
            db.save(path)
            assert path.exists()

            db2 = ProfileDatabase()
            db2.load(path)
            p = db2.lookup("matmul", (128, 128), "fp16")
            assert p is not None
            assert p.mean_latency_us == 320.0
            assert p.sample_count == 1

    def test_load_nonexistent_file(self):
        """Loading from non-existent file is a no-op."""
        db = ProfileDatabase()
        db.load("/nonexistent/path.json")
        assert db.all_profiles() == ()


class TestProfileAwareScheduling:
    """v0.18 Phase 3 B1: compute_scheduling_instructions with profile data."""

    def test_no_profile_uses_default(self):
        """Without profile_db, uses default block_dim."""
        from sonata.pipeline import DispatchPlan, RegionDispatchResult, compute_scheduling_instructions
        dispatch = DispatchPlan(
            results=(RegionDispatchResult(region_id="r0", status="static", action="optimized"),),
            optimized_count=1,
        )
        inst = compute_scheduling_instructions(dispatch)
        assert inst[0].block_dim == 32
        assert "optimized" in inst[0].reason

    def test_high_latency_increases_block_dim(self):
        """High latency profile → higher block_dim for more parallelism."""
        from sonata.pipeline import DispatchPlan, RegionDispatchResult, compute_scheduling_instructions
        from sonata.profile import ProfileDatabase
        db = ProfileDatabase()
        db.record("matmul", (128, 128), "fp16", "aic", 2000.0)  # >1ms
        dispatch = DispatchPlan(
            results=(RegionDispatchResult(region_id="r0", status="static", action="optimized"),),
            optimized_count=1,
        )
        inst = compute_scheduling_instructions(dispatch, profile_db=db)
        assert inst[0].block_dim == 64  # min(32*2, 64)
        assert "profile-informed" in inst[0].reason

    def test_low_latency_decreases_block_dim(self):
        """Low latency profile → lower block_dim (less overhead)."""
        from sonata.pipeline import DispatchPlan, RegionDispatchResult, compute_scheduling_instructions
        from sonata.profile import ProfileDatabase
        db = ProfileDatabase()
        db.record("add", (1024,), "fp16", "aiv", 50.0)  # <100us
        dispatch = DispatchPlan(
            results=(RegionDispatchResult(region_id="r0", status="static", action="optimized"),),
            optimized_count=1,
        )
        inst = compute_scheduling_instructions(dispatch, profile_db=db)
        assert inst[0].block_dim == 16  # max(32//2, 4)
        assert "profile-informed" in inst[0].reason

    def test_dynamic_region_unaffected_by_profile(self):
        """Dynamic regions always use fallback_block_dim regardless of profile."""
        from sonata.pipeline import DispatchPlan, RegionDispatchResult, compute_scheduling_instructions
        from sonata.profile import ProfileDatabase
        db = ProfileDatabase()
        db.record("matmul", (128, 128), "fp16", "aic", 2000.0)
        dispatch = DispatchPlan(
            results=(RegionDispatchResult(region_id="r0", status="dynamic", action="fallback"),),
            fallback_count=1,
        )
        inst = compute_scheduling_instructions(dispatch, profile_db=db)
        assert inst[0].block_dim == 1
        assert "fallback" in inst[0].reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
