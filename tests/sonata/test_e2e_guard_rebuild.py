# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end tests: guard check → STALE rebuild → execution flow.

v0.22 Phase 1 A2: Verifies the full guard lifecycle:
1. Shape change → STALE → plan handle rebuild → Score fingerprint unchanged
2. Hard guard → ALL_FAILED → execution skipped
3. guard_status written to sonata_plan.json
"""

import json
import tempfile
import warnings
from pathlib import Path

import pytest

from sonata.guard import GUARD_SEVERITY_HARD, GUARD_SEVERITY_SOFT
from sonata.plan_handle import GuardStatus, PlanHandle
from sonata.pipeline import (
    SonataAnalysisResult,
    check_guards_at_runtime,
    update_region_guard_status,
)
from sonata.score import RuntimeTarget, Score, ShapeAssumption, Task
from sonata.serialization import score_fingerprint


def _make_score(severity, symbol="batch", dims=(32,)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return Score(
            name="test",
            runtime_target=RuntimeTarget(),
            tasks=(Task(task_id=0, func_id=0, core_type="aic"),),
            shape_assumptions=(ShapeAssumption(symbol=symbol, dims=dims, severity=severity),),
        )


class TestE2EGuardRebuild:
    """End-to-end: guard check → STALE → rebuild → Score fingerprint unchanged."""

    def test_shape_change_stale_fingerprint_unchanged(self):
        """Shape change → STALE, Score fingerprint unchanged (no replan needed)."""
        score = _make_score(GUARD_SEVERITY_SOFT)
        fp_before = score_fingerprint(score)

        result = SonataAnalysisResult(
            eligible=True, score=score, region_statuses={"r0": "static"},
        )
        guard_results = check_guards_at_runtime(result, {"batch": [64]})
        assert guard_results[0].guard_status == "stale"

        # Fingerprint should NOT change
        fp_after = score_fingerprint(score)
        assert fp_before == fp_after

    def test_stale_triggers_plan_handle_rebuild(self):
        """STALE → plan handle can be rebuilt from Score with correct fingerprint."""
        score = _make_score(GUARD_SEVERITY_SOFT)
        original_ph = PlanHandle.from_score(score)

        result = SonataAnalysisResult(
            eligible=True, score=score, region_statuses={"r0": "static"},
        )
        guard_results = check_guards_at_runtime(result, {"batch": [64]})
        assert guard_results[0].guard_status == "stale"

        # Rebuild plan handle — fingerprint should match original
        new_ph = PlanHandle.from_score(score)
        assert new_ph.score_fingerprint == original_ph.score_fingerprint

    def test_hard_guard_all_failed(self):
        """Hard guard violation → ALL_FAILED, execution should be skipped."""
        score = _make_score(GUARD_SEVERITY_HARD)
        result = SonataAnalysisResult(
            eligible=True, score=score, region_statuses={"r0": "static"},
        )
        guard_results = check_guards_at_runtime(result, {"batch": [64]})
        assert guard_results[0].guard_status == "all_failed"

    def test_guard_status_written_to_plan_json(self):
        """guard_status update written to sonata_plan.json."""
        score = _make_score(GUARD_SEVERITY_SOFT)
        ph = PlanHandle.from_score(score)
        result = SonataAnalysisResult(
            eligible=True, score=score, plan_handle=ph,
            region_statuses={"r0": "static"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write initial plan
            plan_path = Path(tmpdir) / "sonata_plan.json"
            result.save(plan_path)

            # Check guards
            guard_results = check_guards_at_runtime(result, {"batch": [64]})
            assert guard_results[0].guard_status == "stale"

            # Update guard status
            status_map = update_region_guard_status(ph, guard_results)
            assert "r0" in status_map

            # Write updated plan
            data = json.loads(plan_path.read_text())
            data["runtime_guard_status"] = {
                k: v.value for k, v in status_map.items()
            }
            plan_path.write_text(json.dumps(data, indent=2, sort_keys=True))

            # Verify written
            updated = json.loads(plan_path.read_text())
            assert "runtime_guard_status" in updated
            assert updated["runtime_guard_status"]["r0"] == "stale"

    def test_all_satisfied_no_action(self):
        """All guards satisfied → no action needed."""
        score = _make_score(GUARD_SEVERITY_SOFT)
        result = SonataAnalysisResult(
            eligible=True, score=score, region_statuses={"r0": "static"},
        )
        guard_results = check_guards_at_runtime(result, {"batch": [32]})
        assert guard_results[0].guard_status == "all_satisfied"
        assert len(guard_results[0].violated_guards) == 0

    def test_mixed_severity_stale(self):
        """Hard satisfied + soft failed → STALE (not ALL_FAILED)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            score = Score(
                name="test",
                runtime_target=RuntimeTarget(),
                tasks=(Task(task_id=0, func_id=0, core_type="aic"),),
                shape_assumptions=(
                    ShapeAssumption(symbol="batch", dims=(32,), severity=GUARD_SEVERITY_HARD),
                    ShapeAssumption(symbol="seq", dims=(128,), severity=GUARD_SEVERITY_SOFT),
                ),
            )
        result = SonataAnalysisResult(
            eligible=True, score=score, region_statuses={"r0": "static"},
        )
        # batch=32 (hard satisfied), seq=256 (soft violated)
        guard_results = check_guards_at_runtime(result, {"batch": [32], "seq": [256]})
        assert guard_results[0].guard_status == "stale"
        assert "seq" in guard_results[0].violated_guards
        assert "batch" not in guard_results[0].violated_guards


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
