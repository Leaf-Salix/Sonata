"""Tests for the deserialization module."""

import json

import pytest

from sonata.deserialization import (
    DeserializationError,
    eligibility_result_from_dict,
    eligibility_result_from_json,
    plan_handle_from_dict,
    plan_handle_from_json,
    score_from_dict,
    score_from_json,
)
from sonata.guard import GUARD_SEVERITY_HARD
from sonata.plan_handle import PlanHandle
from sonata.score import (
    Dependency,
    EligibilityResult,
    FallbackReason,
    RuntimeTarget,
    Score,
    ShapeAssumption,
    Task,
    raw_runtime_target,
)
from sonata.serialization import (
    eligibility_result_to_dict,
    plan_handle_to_dict,
    score_fingerprint,
    score_to_dict,
    score_to_json,
)


def _make_score() -> Score:
    return Score(
        name="roundtrip_graph",
        runtime_target=RuntimeTarget(
            runtime="tensormap_and_ringbuffer",
            function_name="aicpu_orchestration_entry",
            aicpu_thread_num=4,
            config_comment=("# line1", "# line2"),
        ),
        tasks=(
            Task(task_id=0, func_id=10, core_type="aic", name="func_a",
                 args=("x", "y"), arg_directions=("input", "output"),
                 arg_storage_keys=("buf:x", "buf:y")),
            Task(task_id=1, func_id=20, core_type="aiv", name="func_b",
                 args=("y", "z"), arg_directions=("input", "output"),
                 arg_storage_keys=("buf:y", "buf:z")),
        ),
        dependencies=(
            Dependency(producer=0, consumer=1, kind="data"),
            Dependency(producer=0, consumer=1, kind="storage"),
        ),
        shape_assumptions=(
            ShapeAssumption(symbol="N", dims=(128, 64), severity=GUARD_SEVERITY_HARD),
            ShapeAssumption(symbol="M", dims=(32,), severity=GUARD_SEVERITY_HARD),
        ),
        metadata={"extractor": "structural_v0", "dependency_policy": "dataflow_v0"},
    )


class TestScoreRoundTrip:
    def test_dict_roundtrip(self):
        original = _make_score()
        data = score_to_dict(original)
        restored = score_from_dict(data)
        assert restored.name == original.name
        assert raw_runtime_target(restored) == raw_runtime_target(original)
        assert restored.tasks == original.tasks
        assert restored.dependencies == original.dependencies
        assert restored.shape_assumptions == original.shape_assumptions
        assert restored.metadata == original.metadata

    def test_json_roundtrip(self):
        original = _make_score()
        text = score_to_json(original)
        restored = score_from_json(text)
        assert score_to_dict(restored) == score_to_dict(original)

    def test_double_roundtrip_stable(self):
        original = _make_score()
        data1 = score_to_dict(original)
        restored1 = score_from_dict(data1)
        data2 = score_to_dict(restored1)
        assert data1 == data2

    def test_fingerprint_preserved(self):
        original = _make_score()
        fp_original = score_fingerprint(original)
        restored = score_from_dict(score_to_dict(original))
        fp_restored = score_fingerprint(restored)
        assert fp_original == fp_restored

    def test_minimal_score(self):
        score = Score(name="minimal", runtime_target=RuntimeTarget())
        restored = score_from_dict(score_to_dict(score))
        assert score_to_dict(restored) == score_to_dict(score)

    def test_dependency_kind_preserved(self):
        from sonata.score import DependencyKind
        score = Score(
            name="kinds", runtime_target=RuntimeTarget(),
            tasks=(Task(task_id=0, func_id=0, core_type="aic"),
                   Task(task_id=1, func_id=1, core_type="aiv")),
            dependencies=(
                Dependency(producer=0, consumer=1, kind=DependencyKind.WAR),
                Dependency(producer=0, consumer=1, kind=DependencyKind.ORDERING),
            ),
        )
        restored = score_from_dict(score_to_dict(score))
        assert restored.dependencies[0].kind == DependencyKind.WAR
        assert restored.dependencies[1].kind == DependencyKind.ORDERING

    def test_dependency_kind_backward_compat(self):
        """Old dicts with plain string 'kind' still deserialize."""
        from sonata.score import DependencyKind
        data = {
            "schema_version": 1,
            "name": "compat",
            "runtime_target": {"runtime": "host_build_graph", "function_name": "f", "aicpu_thread_num": 1},
            "tasks": [],
            "dependencies": [
                {"producer": 0, "consumer": 1, "kind": "storage"},
            ],
            "shape_assumptions": [],
            "metadata": {},
        }
        restored = score_from_dict(data)
        assert restored.dependencies[0].kind == DependencyKind.STORAGE


class TestPlanHandleRoundTrip:
    def test_dict_roundtrip(self):
        score = _make_score()
        plan = PlanHandle.from_score(score)
        data = plan_handle_to_dict(plan)
        restored = plan_handle_from_dict(data)
        assert restored.score_fingerprint == plan.score_fingerprint
        assert restored.runtime_target == plan.runtime_target
        assert restored.source_adapter == plan.source_adapter
        assert restored.func_registry.entries == plan.func_registry.entries
        assert restored.arg_bindings == plan.arg_bindings

    def test_json_roundtrip(self):
        score = _make_score()
        plan = PlanHandle.from_score(score)
        from sonata.serialization import plan_handle_to_json
        text = plan_handle_to_json(plan)
        restored = plan_handle_from_json(text)
        assert restored.score_fingerprint == plan.score_fingerprint
        assert restored.source_adapter == plan.source_adapter


class TestEligibilityResultRoundTrip:
    def test_eligible_roundtrip(self):
        score = _make_score()
        result = EligibilityResult.accept(score)
        data = eligibility_result_to_dict(result)
        restored = eligibility_result_from_dict(data)
        assert restored.eligible is True
        assert restored.score is not None
        assert restored.score.name == score.name

    def test_rejected_roundtrip(self):
        result = EligibilityResult.reject(
            FallbackReason(code="no_tasks", message="score has no tasks", severity="error")
        )
        data = eligibility_result_to_dict(result)
        restored = eligibility_result_from_dict(data)
        assert restored.eligible is False
        assert len(restored.reason_details) == 1
        assert restored.reason_details[0].code == "no_tasks"

    def test_accept_with_warnings_roundtrip(self):
        score = _make_score()
        result = EligibilityResult.accept_with_warnings(
            score, FallbackReason(code="low_coverage", message="coverage 60%", severity="warning")
        )
        data = eligibility_result_to_dict(result)
        restored = eligibility_result_from_dict(data)
        assert restored.eligible is True
        assert restored.has_warnings()
        assert restored.reason_details[0].severity == "warning"

    def test_json_roundtrip(self):
        score = _make_score()
        result = EligibilityResult.accept(score)
        data = eligibility_result_to_dict(result)
        text = json.dumps(data, sort_keys=True)
        restored = eligibility_result_from_json(text)
        assert restored.eligible is True


class TestDeserializationErrors:
    def test_score_missing_name(self):
        data = score_to_dict(_make_score())
        del data["name"]
        with pytest.raises(DeserializationError, match="name"):
            score_from_dict(data)

    def test_score_wrong_type(self):
        with pytest.raises(DeserializationError, match="must be a dict"):
            score_from_dict("not a dict")

    def test_score_schema_mismatch(self):
        data = score_to_dict(_make_score())
        data["schema_version"] = 999
        with pytest.raises(DeserializationError, match="schema version mismatch"):
            score_from_dict(data)

    def test_score_task_missing_field(self):
        data = score_to_dict(_make_score())
        del data["tasks"][0]["task_id"]
        with pytest.raises(DeserializationError, match="task_id"):
            score_from_dict(data)

    def test_plan_handle_missing_fingerprint(self):
        score = _make_score()
        data = plan_handle_to_dict(PlanHandle.from_score(score))
        del data["score_fingerprint"]
        with pytest.raises(DeserializationError, match="score_fingerprint"):
            plan_handle_from_dict(data)

    def test_eligibility_missing_eligible(self):
        data = eligibility_result_to_dict(EligibilityResult.accept(_make_score()))
        del data["eligible"]
        with pytest.raises(DeserializationError, match="eligible"):
            eligibility_result_from_dict(data)

    def test_invalid_json(self):
        with pytest.raises(DeserializationError, match="invalid JSON"):
            score_from_json("{not valid json")

    def test_json_not_dict(self):
        with pytest.raises(DeserializationError, match="must decode to a dict"):
            score_from_json("[1, 2, 3]")

    def test_tasks_not_list(self):
        data = score_to_dict(_make_score())
        data["tasks"] = "wrong"
        with pytest.raises(DeserializationError, match="tasks must be a list"):
            score_from_dict(data)

    def test_dependency_wrong_type(self):
        data = score_to_dict(_make_score())
        data["dependencies"] = [{"producer": "bad", "consumer": 1}]
        with pytest.raises(DeserializationError, match="producer"):
            score_from_dict(data)


class TestPlanHandleGuardStatusDeserialization:
    """Bug fix: plan_handle_from_dict must read guard_status and critical_guards."""

    def test_guard_status_round_trip(self):
        """guard_status survives serialization → deserialization."""
        from sonata.plan_handle import PlanHandle, GuardStatus
        from sonata.score import Score, RuntimeTarget, Task
        from sonata.serialization import plan_handle_to_dict
        from sonata.deserialization import plan_handle_from_dict

        score = Score(name="t", runtime_target=RuntimeTarget(),
                      tasks=(Task(task_id=0, func_id=0, core_type="aic"),))
        ph = PlanHandle.from_score(score)
        ph_failed = PlanHandle(
            score_fingerprint=ph.score_fingerprint,
            runtime_target=ph.runtime_target,
            source_adapter=ph.source_adapter,
            func_registry=ph.func_registry,
            guard_status=GuardStatus.ALL_FAILED,
        )
        d = plan_handle_to_dict(ph_failed)
        restored = plan_handle_from_dict(d)
        assert restored.guard_status == GuardStatus.ALL_FAILED

    def test_stale_guard_status_round_trip(self):
        """STALE guard_status survives round-trip."""
        from sonata.plan_handle import PlanHandle, GuardStatus
        from sonata.score import Score, RuntimeTarget, Task
        from sonata.serialization import plan_handle_to_dict
        from sonata.deserialization import plan_handle_from_dict

        score = Score(name="t", runtime_target=RuntimeTarget(),
                      tasks=(Task(task_id=0, func_id=0, core_type="aic"),))
        ph = PlanHandle.from_score(score)
        ph_stale = PlanHandle(
            score_fingerprint=ph.score_fingerprint,
            runtime_target=ph.runtime_target,
            source_adapter=ph.source_adapter,
            func_registry=ph.func_registry,
            guard_status=GuardStatus.STALE,
        )
        d = plan_handle_to_dict(ph_stale)
        restored = plan_handle_from_dict(d)
        assert restored.guard_status == GuardStatus.STALE

    def test_critical_guards_not_list_raises(self):
        """Non-list critical_guards should raise DeserializationError."""
        from sonata.plan_handle import PlanHandle, GuardStatus
        from sonata.score import Score, RuntimeTarget, Task
        from sonata.serialization import plan_handle_to_dict

        score = Score(name="t", runtime_target=RuntimeTarget(),
                      tasks=(Task(task_id=0, func_id=0, core_type="aic"),))
        ph = PlanHandle.from_score(score)
        d = plan_handle_to_dict(ph)
        d["critical_guards"] = "not_a_list"
        with pytest.raises(DeserializationError, match="critical_guards must be a list"):
            plan_handle_from_dict(d)

    def test_critical_guards_bad_item_raises(self):
        """Non-dict items in critical_guards should raise DeserializationError."""
        from sonata.plan_handle import PlanHandle
        from sonata.score import Score, RuntimeTarget, Task
        from sonata.serialization import plan_handle_to_dict

        score = Score(name="t", runtime_target=RuntimeTarget(),
                      tasks=(Task(task_id=0, func_id=0, core_type="aic"),))
        ph = PlanHandle.from_score(score)
        d = plan_handle_to_dict(ph)
        d["critical_guards"] = ["not_a_dict"]
        with pytest.raises(DeserializationError, match="expected dict"):
            plan_handle_from_dict(d)
