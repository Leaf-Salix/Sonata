import json

from sonata import RuntimeTarget, Score, Task, score_fingerprint, score_to_dict, score_to_json


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
        metadata={"nodep_args": ({"task_id": 0, "arg": "x"},)},
    )

    data = score_to_dict(score)

    assert data["schema_version"] == 1
    assert data["runtime_target"]["aicpu_thread_num"] is None
    assert data["tasks"][0]["arg_storage_keys"] == ["param:x", "alloc:out", None]
    assert data["metadata"] == {"nodep_args": [{"arg": "x", "task_id": 0}]}


def test_score_to_json_is_valid_sorted_json() -> None:
    score = Score(
        name="empty",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_empty_graph"),
    )

    text = score_to_json(score)

    assert json.loads(text)["name"] == "empty"
    assert text.splitlines()[1].strip() == '"dependencies": [],'


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
