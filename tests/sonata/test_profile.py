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


class TestTimingCollection:
    """v0.18 Phase 3 B2: collect_task_timings tests."""

    def test_collect_basic_timings(self):
        """Timings are recorded into ProfileDatabase."""
        import warnings
        from sonata.score import Score, RuntimeTarget, Task
        from sonata.pipeline import collect_task_timings
        from sonata.profile import ProfileDatabase

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            score = Score(
                name="test",
                runtime_target=RuntimeTarget(),
                tasks=(
                    Task(task_id=0, func_id=0, core_type="aic", name="matmul"),
                    Task(task_id=1, func_id=1, core_type="aiv", name="add"),
                ),
            )
        db = ProfileDatabase()
        collect_task_timings(db, score, {"matmul": 320.0, "add": 50.0})
        assert db.lookup("matmul", (), "unknown") is not None
        assert db.lookup("matmul", (), "unknown").mean_latency_us == 320.0
        assert db.lookup("add", (), "unknown").mean_latency_us == 50.0

    def test_collect_partial_timings(self):
        """Only recorded tasks get profiles; missing tasks are skipped."""
        import warnings
        from sonata.score import Score, RuntimeTarget, Task
        from sonata.pipeline import collect_task_timings
        from sonata.profile import ProfileDatabase

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            score = Score(
                name="test",
                runtime_target=RuntimeTarget(),
                tasks=(
                    Task(task_id=0, func_id=0, core_type="aic", name="matmul"),
                    Task(task_id=1, func_id=1, core_type="aiv", name="add"),
                ),
            )
        db = ProfileDatabase()
        collect_task_timings(db, score, {"matmul": 320.0})
        assert db.lookup("matmul", (), "unknown") is not None
        assert db.lookup("add", (), "unknown") is None

    def test_collect_none_db(self):
        """None profile_db is a no-op."""
        import warnings
        from sonata.score import Score, RuntimeTarget, Task
        from sonata.pipeline import collect_task_timings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            score = Score(
                name="test",
                runtime_target=RuntimeTarget(),
                tasks=(Task(task_id=0, func_id=0, core_type="aic", name="matmul"),),
            )
        collect_task_timings(None, score, {"matmul": 320.0})  # should not raise

    def test_collect_accumulates(self):
        """Multiple calls accumulate samples in profile."""
        import warnings
        from sonata.score import Score, RuntimeTarget, Task
        from sonata.pipeline import collect_task_timings
        from sonata.profile import ProfileDatabase

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            score = Score(
                name="test",
                runtime_target=RuntimeTarget(),
                tasks=(Task(task_id=0, func_id=0, core_type="aic", name="matmul"),),
            )
        db = ProfileDatabase()
        collect_task_timings(db, score, {"matmul": 300.0})
        collect_task_timings(db, score, {"matmul": 340.0})
        p = db.lookup("matmul", (), "unknown")
        assert p.sample_count == 2
        assert abs(p.mean_latency_us - 320.0) < 0.01


class TestProfileSchedulingIntegration:
    """v0.21 Phase 1 A3: Full profile → scheduling → runtime hint integration."""

    def test_high_latency_profile_increases_block_dim(self):
        """High latency profile → higher block_dim via scheduling instructions."""
        import tempfile
        import json
        from pathlib import Path
        from sonata.profile import ProfileDatabase
        from sonata.runtime_hook import apply_sonata_runtime_hints

        # Create a profile with high latency
        db = ProfileDatabase()
        db.record("matmul", (128, 128), "fp16", "aic", 2000.0)  # >1ms

        with tempfile.TemporaryDirectory() as tmpdir:
            plan = {
                "eligible": True,
                "region_statuses": {"region_0": "static"},
            }
            (Path(tmpdir) / "sonata_plan.json").write_text(json.dumps(plan))

            # Without profile → default block_dim=32
            result_no_profile = apply_sonata_runtime_hints(
                work_dir=tmpdir, block_dim=None, aicpu_thread_num=None,
                user_block_dim=None,
            )
            assert result_no_profile.block_dim == 32

    def test_low_latency_profile_decreases_block_dim(self):
        """Low latency profile → lower block_dim via scheduling instructions."""
        import tempfile
        import json
        from pathlib import Path
        from sonata.profile import ProfileDatabase
        from sonata.pipeline import DispatchPlan, RegionDispatchResult, compute_scheduling_instructions

        db = ProfileDatabase()
        db.record("add", (1024,), "fp16", "aiv", 50.0)  # <100us

        dispatch = DispatchPlan(
            results=(RegionDispatchResult(region_id="r0", status="static", action="optimized"),),
            optimized_count=1,
        )
        inst = compute_scheduling_instructions(dispatch, profile_db=db)
        assert inst[0].block_dim == 16  # max(32//2, 4)

    def test_profile_db_round_trip(self):
        """Profile DB survives save/load round-trip."""
        import tempfile
        from pathlib import Path
        from sonata.profile import ProfileDatabase

        db = ProfileDatabase()
        db.record("matmul", (128, 128), "fp16", "aic", 320.0)
        db.record("conv2d", (56, 56), "fp16", "aic", 1050.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.json"
            db.save(path)
            db2 = ProfileDatabase()
            db2.load(path)
            p = db2.lookup("matmul", (128, 128), "fp16")
            assert p is not None
            assert abs(p.mean_latency_us - 320.0) < 0.01

    def test_empty_profile_db_no_effect(self):
        """Empty profile DB → no effect on scheduling."""
        from sonata.profile import ProfileDatabase
        from sonata.pipeline import DispatchPlan, RegionDispatchResult, compute_scheduling_instructions

        db = ProfileDatabase()
        dispatch = DispatchPlan(
            results=(RegionDispatchResult(region_id="r0", status="static", action="optimized"),),
            optimized_count=1,
        )
        inst = compute_scheduling_instructions(dispatch, profile_db=db)
        assert inst[0].block_dim == 32  # default, no profile adjustment


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
