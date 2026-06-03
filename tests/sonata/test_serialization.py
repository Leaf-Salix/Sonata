import json

import sonata.serialization as serialization
from sonata import (
    ELIGIBILITY_RESULT_SCHEMA_VERSION,
    FINGERPRINT_VERSION,
    EligibilityResult,
    RuntimeTarget,
    SCORE_SCHEMA_VERSION,
    Score,
    ShapeAssumption,
    Task,
    eligibility_result_to_dict,
    score_fingerprint,
    score_to_dict,
    score_to_json,
)
from sonata.guard import GUARD_SEVERITY_HARD


def test_score_to_dict_emits_stable_json_like_structure() -> None:
    score = Score(
        name="vector_add_score",
        runtime_target=RuntimeTarget(
            runtime="host_build_graph",
            function_name="build_vector_add_graph",
            aicpu_thread_num=None,
            config_comment=("# host static graph",),
        ),
        tasks=(
            Task(
                task_id=0,
                func_id=0,
                core_type="aiv",
                args=("x", "out", 1),
                arg_directions=("Input", "OutputExisting", "Scalar"),
                arg_storage_keys=("param:x", "alloc:out", None),
                name="kernel",
            ),
        ),
        shape_assumptions=(ShapeAssumption(symbol="x", dims=(64, 32), severity="hard"),),
        metadata={"nodep_args": ({"task_id": 0, "arg": "x"},)},
    )

    data = score_to_dict(score)

    assert data["schema_version"] == SCORE_SCHEMA_VERSION
    assert "fingerprint_version" not in data
    assert data["runtime_target"]["aicpu_thread_num"] is None
    assert data["tasks"][0]["arg_storage_keys"] == ["param:x", "alloc:out", None]
    assert data["shape_assumptions"] == [{"symbol": "x", "dims": [64, 32], "severity": "hard"}]
    assert data["metadata"] == {"nodep_args": [{"arg": "x", "task_id": 0}]}


def test_score_to_json_is_valid_sorted_json() -> None:
    score = Score(
        name="empty",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_empty_graph"),
    )

    text = score_to_json(score)

    assert json.loads(text)["name"] == "empty"
    assert text.splitlines()[1].strip() == '"dependencies": [],'


def test_score_to_json_matches_v1_golden_schema() -> None:
    score = Score(
        name="golden",
        runtime_target=RuntimeTarget(
            runtime="host_build_graph",
            function_name="build_golden_graph",
            aicpu_thread_num=None,
            config_comment=("# generated",),
        ),
        tasks=(
            Task(
                task_id=0,
                func_id=1,
                core_type="aiv",
                args=("x", 4),
                arg_directions=("Input", "Scalar"),
                arg_storage_keys=("param:x", None),
                name="kernel.add",
            ),
        ),
        shape_assumptions=(ShapeAssumption(symbol="x", dims=(16,), severity=GUARD_SEVERITY_HARD),),
        metadata={"dependency_policy": "sequential_v0"},
    )

    assert score_to_json(score) == """{
  "dependencies": [],
  "metadata": {
    "dependency_policy": "sequential_v0"
  },
  "name": "golden",
  "runtime_target": {
    "aicpu_thread_num": null,
    "config_comment": [
      "# generated"
    ],
    "function_name": "build_golden_graph",
    "runtime": "host_build_graph"
  },
  "schema_version": 1,
  "shape_assumptions": [
    {
      "dims": [
        16
      ],
      "severity": "hard",
      "symbol": "x"
    }
  ],
  "tasks": [
    {
      "arg_directions": [
        "Input",
        "Scalar"
      ],
      "arg_storage_keys": [
        "param:x",
        null
      ],
      "args": [
        "x",
        4
      ],
      "core_type": "aiv",
      "func_id": 1,
      "name": "kernel.add",
      "outputs": [],
      "storage_effects": [],
      "task_id": 0
    }
  ]
}"""


def test_eligibility_result_to_dict_emits_structured_fallback_reasons() -> None:
    result = EligibilityResult.reject("tensor.read calls are not supported by initial Sonata eligibility")

    data = eligibility_result_to_dict(result)

    assert data["schema_version"] == ELIGIBILITY_RESULT_SCHEMA_VERSION
    assert data["eligible"] is False
    assert data["score"] is None
    assert data["reasons"] == ["tensor.read calls are not supported by initial Sonata eligibility"]
    assert data["reason_details"] == [
        {
            "code": "tensor_read_calls_are_not_supported_by_initial_sonata_eligibility",
            "message": "tensor.read calls are not supported by initial Sonata eligibility",
            "severity": "error",
        }
    ]


def test_score_fingerprint_ignores_metadata_by_default() -> None:
    base = Score(
        name="same_plan",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_same_plan_graph"),
        tasks=(Task(task_id=0, func_id=0, core_type="aiv", args=("x",), name="kernel"),),
        metadata={"audit": "first"},
    )
    changed_metadata = Score(
        name="same_plan",
        runtime_target=base.runtime_target,
        tasks=base.tasks,
        metadata={"audit": "second"},
    )

    assert score_fingerprint(base) == score_fingerprint(changed_metadata)
    assert score_fingerprint(base, include_metadata=True) != score_fingerprint(changed_metadata, include_metadata=True)


def test_score_fingerprint_ignores_runtime_target_by_default() -> None:
    base = Score(
        name="same_plan",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_one"),
        tasks=(Task(task_id=0, func_id=0, core_type="aiv", args=("x",), name="kernel"),),
        metadata={"audit": "same"},
    )
    changed_runtime = Score(
        name="same_plan",
        runtime_target=RuntimeTarget(
            runtime="tensormap_and_ringbuffer",
            function_name="build_two",
            aicpu_thread_num=4,
            config_comment=("# different runtime contract",),
        ),
        tasks=base.tasks,
        metadata=base.metadata,
    )

    assert score_fingerprint(base) == score_fingerprint(changed_runtime)
    assert score_fingerprint(base, include_metadata=True) == score_fingerprint(
        changed_runtime,
        include_metadata=True,
    )


def test_score_fingerprint_version_changes_hash_payload(monkeypatch) -> None:
    score = Score(
        name="versioned_plan",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_versioned_graph"),
    )

    before = score_fingerprint(score)
    monkeypatch.setattr(serialization, "FINGERPRINT_VERSION", FINGERPRINT_VERSION + 1)

    assert score_fingerprint(score) != before


def test_score_fingerprint_changes_for_plan_semantics() -> None:
    first = Score(
        name="plan",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_plan_graph"),
        tasks=(Task(task_id=0, func_id=0, core_type="aiv", args=("x",), name="kernel"),),
    )
    second = Score(
        name="plan",
        runtime_target=first.runtime_target,
        tasks=(Task(task_id=0, func_id=1, core_type="aiv", args=("x",), name="kernel"),),
    )

    assert score_fingerprint(first) != score_fingerprint(second)


def test_score_fingerprint_changes_for_shape_assumptions() -> None:
    first = Score(
        name="plan",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_plan_graph"),
        shape_assumptions=(ShapeAssumption(symbol="x", dims=(64, 32), severity="hard"),),
    )
    second = Score(
        name="plan",
        runtime_target=first.runtime_target,
        shape_assumptions=(ShapeAssumption(symbol="x", dims=(128, 32), severity="hard"),),
    )

    assert score_fingerprint(first) != score_fingerprint(second)


def test_plan_handle_region_guard_status_roundtrip() -> None:
    """region_guard_status survives serialization -> deserialization."""
    from sonata.plan_handle import GuardStatus, PlanHandle, RuntimeTarget
    from sonata.serialization import plan_handle_to_dict
    from sonata.deserialization import plan_handle_from_dict

    rt = RuntimeTarget(runtime="host_build_graph", function_name="f", aicpu_thread_num=1)
    ph = PlanHandle(
        score_fingerprint="abc123",
        runtime_target=rt,
        source_adapter="test",
        region_guard_status={
            "root": GuardStatus.ALL_SATISFIED,
            "root.child[0]": GuardStatus.PARTIAL_FAILED,
            "root.child[1]": GuardStatus.ALL_FAILED,
        },
    )
    data = plan_handle_to_dict(ph)
    assert data["region_guard_status"] == {
        "root": "all_satisfied",
        "root.child[0]": "partial_failed",
        "root.child[1]": "all_failed",
    }

    restored = plan_handle_from_dict(data)
    assert restored.region_guard_status == ph.region_guard_status


class TestStorageEffectSerialization:
    """v0.17 Phase 3 B2: StorageEffect round-trip tests."""

    def test_round_trip_with_effects(self):
        """Score with storage_effects survives serialization round-trip."""
        from sonata.score import StorageEffect
        from sonata.deserialization import score_from_dict
        effects = (
            StorageEffect(buffer_id="buf_x", kind="read"),
            StorageEffect(buffer_id="buf_y", kind="write"),
        )
        score = Score(
            name="test",
            runtime_target=RuntimeTarget(),
            tasks=(Task(task_id=0, func_id=0, core_type="aic",
                        args=("x", "y"), arg_directions=("input", "output"),
                        arg_storage_keys=("buf:x", "buf:y"),
                        storage_effects=effects),),
        )
        d = score_to_dict(score)
        assert d["tasks"][0]["storage_effects"] == [
            {"buffer_id": "buf_x", "kind": "read"},
            {"buffer_id": "buf_y", "kind": "write"},
        ]
        restored = score_from_dict(d)
        assert restored.tasks[0].storage_effects == effects

    def test_round_trip_none_effects(self):
        """Score with storage_effects=None survives round-trip as None."""
        from sonata.deserialization import score_from_dict
        score = Score(
            name="test",
            runtime_target=RuntimeTarget(),
            tasks=(Task(task_id=0, func_id=0, core_type="aic",
                        storage_effects=None),),
        )
        d = score_to_dict(score)
        assert d["tasks"][0]["storage_effects"] == []
        restored = score_from_dict(d)
        assert restored.tasks[0].storage_effects is None

    def test_round_trip_empty_effects(self):
        """Score with storage_effects=() survives round-trip as None."""
        from sonata.deserialization import score_from_dict
        score = Score(
            name="test",
            runtime_target=RuntimeTarget(),
            tasks=(Task(task_id=0, func_id=0, core_type="aic",
                        storage_effects=()),),
        )
        d = score_to_dict(score)
        restored = score_from_dict(d)
        assert restored.tasks[0].storage_effects is None
