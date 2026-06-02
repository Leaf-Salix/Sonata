# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
from sonata import (
    Dependency,
    RuntimeTarget,
    Score,
    ShapeAssumption,
    Task,
)
from sonata.score import EligibilityResult, FallbackReason


def test_runtime_target_defaults_to_current_tensormap_contract() -> None:
    target = RuntimeTarget()

    assert target.runtime == "tensormap_and_ringbuffer"
    assert target.function_name == "aicpu_orchestration_entry"
    assert target.aicpu_thread_num == 4


def test_score_counts_tasks_and_dependencies() -> None:
    score = Score(
        name="vector_add_score",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_vector_add_graph"),
        tasks=(
            Task(task_id=0, func_id=0, core_type="aiv", args=("a", "b", "c"), name="add"),
            Task(task_id=1, func_id=1, core_type="aiv", args=("c", "out"), name="store"),
        ),
        dependencies=(Dependency(producer=0, consumer=1),),
        shape_assumptions=(ShapeAssumption(symbol="a", dims=(1024,), severity="hard"),),
    )

    assert score.task_count() == 2
    assert score.dependency_count() == 1
    assert score.shape_assumptions[0].dims == (1024,)


def test_score_validate_accepts_consistent_score() -> None:
    score = Score(
        name="valid",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_valid_graph"),
        tasks=(
            Task(task_id=0, func_id=0, core_type="aiv"),
            Task(task_id=1, func_id=1, core_type="aic"),
        ),
        dependencies=(Dependency(producer=0, consumer=1),),
        shape_assumptions=(ShapeAssumption(symbol="x", dims=(16, 32), severity="hard"),),
    )

    result = score.validate()

    assert result.eligible
    assert result.score is score
    assert result.reasons == ()


def test_score_validate_rejects_unknown_dependency_target() -> None:
    score = Score(
        name="invalid_dep",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_invalid_dep_graph"),
        tasks=(Task(task_id=0, func_id=0, core_type="aiv"),),
        dependencies=(Dependency(producer=0, consumer=2),),
    )

    result = score.validate()

    assert not result.eligible
    assert result.score is None
    assert result.reasons == ("dependency consumer is unknown: 2",)
    assert [(reason.code, reason.message, reason.severity) for reason in result.reason_details] == [
        ("score_validation_failed", "dependency consumer is unknown: 2", "error")
    ]


def test_score_validate_rejects_dependency_cycle() -> None:
    score = Score(
        name="cyclic",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_cyclic_graph"),
        tasks=(
            Task(task_id=0, func_id=0, core_type="aiv"),
            Task(task_id=1, func_id=1, core_type="aiv"),
            Task(task_id=2, func_id=2, core_type="aiv"),
        ),
        dependencies=(
            Dependency(producer=0, consumer=1),
            Dependency(producer=1, consumer=2),
            Dependency(producer=2, consumer=0),
        ),
    )

    result = score.validate()

    assert not result.eligible
    assert result.reasons == ("dependency graph must be acyclic, found cycle: 0 -> 1 -> 2 -> 0",)


def test_score_validate_accumulates_rejection_reasons() -> None:
    score = Score(
        name="",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_invalid_graph"),
        tasks=(
            Task(task_id=0, func_id=0, core_type="aiv"),
            Task(task_id=0, func_id=-1, core_type="gpu"),
        ),
        dependencies=(Dependency(producer=0, consumer=0),),
        shape_assumptions=(ShapeAssumption(symbol="", dims=(4, -1), severity="hard"),),
    )

    result = score.validate()

    assert not result.eligible
    assert result.reasons == (
        "score name must not be empty",
        "task ids must be unique",
        "task 0 func_id must be non-negative",
        "task 0 has unsupported core_type: gpu",
        "dependency cannot be a self-edge: 0",
        "shape assumption symbol must not be empty",
    )


def test_score_validate_rejects_task_arg_direction_arity_mismatch() -> None:
    score = Score(
        name="invalid_task_dirs",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_invalid_graph"),
        tasks=(Task(task_id=0, func_id=0, core_type="aiv", args=("x", "out"), arg_directions=("Input",)),),
    )

    result = score.validate()

    assert not result.eligible
    assert result.reasons == ("task 0 arg_directions size 1 does not match args size 2",)


def test_score_validate_rejects_task_arg_storage_key_arity_mismatch() -> None:
    score = Score(
        name="invalid_task_storage_keys",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_invalid_graph"),
        tasks=(Task(task_id=0, func_id=0, core_type="aiv", args=("x", "out"), arg_storage_keys=("x",)),),
    )

    result = score.validate()

    assert not result.eligible
    assert result.reasons == ("task 0 arg_storage_keys size 1 does not match args size 2",)


def test_score_validate_rejects_duplicate_shape_assumption_symbols() -> None:
    score = Score(
        name="duplicate_shapes",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_invalid_graph"),
        shape_assumptions=(
            ShapeAssumption(symbol="x", dims=(16,), severity="hard"),
            ShapeAssumption(symbol="x", dims=(32,), severity="hard"),
        ),
    )

    result = score.validate()

    assert not result.eligible
    assert result.reasons == ("shape assumption symbol must be unique: x",)


def test_score_validate_does_not_repeat_dim_errors_for_duplicate_shape_symbols() -> None:
    score = Score(
        name="duplicate_bad_shapes",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_invalid_graph"),
        shape_assumptions=(
            ShapeAssumption(symbol="x", dims=(-1,), severity="hard"),
            ShapeAssumption(symbol="x", dims=(-1,), severity="hard"),
        ),
    )

    result = score.validate()

    assert not result.eligible
    assert result.reasons == (
        "shape assumption x has negative dimension",
        "shape assumption symbol must be unique: x",
    )


def test_score_validate_rejects_duplicate_empty_shape_assumption_symbols() -> None:
    score = Score(
        name="duplicate_empty_shapes",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_invalid_graph"),
        shape_assumptions=(
            ShapeAssumption(symbol="", dims=(16,), severity="hard"),
            ShapeAssumption(symbol="", dims=(32,), severity="hard"),
        ),
    )

    result = score.validate()

    assert not result.eligible
    assert result.reasons == (
        "shape assumption symbol must not be empty",
        "shape assumption symbol must be unique: <empty>",
    )


def test_score_validate_skips_dim_validation_for_empty_shape_symbol() -> None:
    score = Score(
        name="empty_shape_symbol",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_invalid_graph"),
        shape_assumptions=(ShapeAssumption(symbol="", dims=("n",), severity="hard"),),
    )

    result = score.validate()

    assert not result.eligible
    assert result.reasons == ("shape assumption symbol must not be empty",)


def test_score_validate_rejects_zero_shape_assumption_dimension() -> None:
    score = Score(
        name="zero_shape_dim",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_invalid_graph"),
        shape_assumptions=(ShapeAssumption(symbol="x", dims=(0, 16), severity="hard"),),
    )

    result = score.validate()

    assert not result.eligible
    assert result.reasons == ("shape assumption x has zero dimension",)


def test_score_validate_rejects_non_integer_shape_assumption_dimension() -> None:
    score = Score(
        name="symbolic_shape_dim",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_invalid_graph"),
        shape_assumptions=(ShapeAssumption(symbol="x", dims=("n", 16), severity="hard"),),
    )

    result = score.validate()

    assert not result.eligible
    assert result.reasons == ("shape assumption x has non-integer dimension",)


def test_score_validate_rejects_bool_shape_assumption_dimension() -> None:
    score = Score(
        name="bool_shape_dim",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_invalid_graph"),
        shape_assumptions=(ShapeAssumption(symbol="x", dims=(True, 16), severity="hard"),),
    )

    result = score.validate()

    assert not result.eligible
    assert result.reasons == ("shape assumption x has non-integer dimension",)


def test_score_validate_reports_each_shape_dim_error_category() -> None:
    score = Score(
        name="multi_bad_shape_dim",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_invalid_graph"),
        shape_assumptions=(ShapeAssumption(symbol="x", dims=(0, -1, "n", True), severity="hard"),),
    )

    result = score.validate()

    assert not result.eligible
    assert result.reasons == (
        "shape assumption x has zero dimension",
        "shape assumption x has negative dimension",
        "shape assumption x has non-integer dimension",
    )


class TestAcceptWithWarnings:
    def test_returns_eligible_with_warning_details(self) -> None:
        score = Score(name="valid", runtime_target=RuntimeTarget())
        result = EligibilityResult.accept_with_warnings(score, "dim x assumed static")

        assert result.eligible
        assert result.score is score
        assert len(result.reason_details) == 1
        assert result.reason_details[0].severity == "warning"
        assert result.reason_details[0].message == "dim x assumed static"

    def test_preserves_score_reference(self) -> None:
        score = Score(name="valid", runtime_target=RuntimeTarget())
        result = EligibilityResult.accept_with_warnings(score, "w1", "w2")

        assert result.score is score
        assert len(result.reason_details) == 2

    def test_with_empty_warnings(self) -> None:
        score = Score(name="valid", runtime_target=RuntimeTarget())
        result = EligibilityResult.accept_with_warnings(score)

        assert result.eligible
        assert result.reason_details == ()

    def test_rejects_none_score(self) -> None:
        with pytest.raises(ValueError, match="non-None score"):
            EligibilityResult.accept_with_warnings(None, "warning")


class TestHasErrorsHasWarnings:
    def test_has_errors_true_on_rejection(self) -> None:
        score = Score(name="", runtime_target=RuntimeTarget())
        result = score.validate()

        assert not result.eligible
        assert result.has_errors()
        assert not result.has_warnings()

    def test_has_errors_false_on_accept(self) -> None:
        score = Score(name="valid", runtime_target=RuntimeTarget())
        result = score.validate()

        assert result.eligible
        assert not result.has_errors()
        assert not result.has_warnings()

    def test_has_warnings_true_on_accept_with_warnings(self) -> None:
        score = Score(name="valid", runtime_target=RuntimeTarget())
        result = EligibilityResult.accept_with_warnings(score, "low coverage")

        assert result.eligible
        assert result.has_warnings()
        assert not result.has_errors()

    def test_has_errors_false_on_empty_reason_details(self) -> None:
        result = EligibilityResult(eligible=True)

        assert not result.has_errors()
        assert not result.has_warnings()


class TestIsStaticShapeDim:
    def test_positive_int_returns_true(self) -> None:
        from sonata.score import is_static_shape_dim

        assert is_static_shape_dim(1) is True
        assert is_static_shape_dim(64) is True
        assert is_static_shape_dim(100000) is True

    def test_zero_returns_false(self) -> None:
        from sonata.score import is_static_shape_dim

        assert is_static_shape_dim(0) is False

    def test_negative_returns_false(self) -> None:
        from sonata.score import is_static_shape_dim

        assert is_static_shape_dim(-1) is False
        assert is_static_shape_dim(-100) is False

    def test_bool_returns_false(self) -> None:
        from sonata.score import is_static_shape_dim

        assert is_static_shape_dim(True) is False
        assert is_static_shape_dim(False) is False

    def test_non_int_returns_false(self) -> None:
        from sonata.score import is_static_shape_dim

        assert is_static_shape_dim("n") is False
        assert is_static_shape_dim(3.14) is False
        assert is_static_shape_dim(None) is False


class TestTaskMultiOutput:
    """Tests for Task.outputs field (v0.11 Phase 3 A1-A3)."""

    def test_task_multi_output_construction(self):
        """Task with multiple outputs stores them correctly."""
        from sonata.score import Task
        t = Task(task_id=0, func_id=1, core_type="aicore",
                 outputs=("buf_0", "buf_1", "buf_2"))
        assert t.outputs == ("buf_0", "buf_1", "buf_2")

    def test_task_single_output(self):
        """Single-output task works as before."""
        from sonata.score import Task
        t = Task(task_id=0, func_id=1, core_type="aicore",
                 outputs=("buf_0",))
        assert t.outputs == ("buf_0",)

    def test_task_outputs_backward_compat(self):
        """Old Task without outputs gets empty tuple (default)."""
        from sonata.score import Task
        t = Task(task_id=0, func_id=1, core_type="aicore")
        assert t.outputs == ()

    def test_task_outputs_roundtrip(self):
        """outputs survives serialization -> deserialization."""
        from sonata.score import Task, Score, RuntimeTarget
        from sonata.serialization import score_to_dict
        from sonata.deserialization import score_from_dict

        rt = RuntimeTarget(runtime="host_build_graph", function_name="f", aicpu_thread_num=1)
        task = Task(task_id=0, func_id=1, core_type="aicore",
                    outputs=("out_a", "out_b"))
        score = Score(name="test", runtime_target=rt, tasks=(task,),
                      dependencies=(), shape_assumptions=())
        data = score_to_dict(score)
        assert data["tasks"][0]["outputs"] == ["out_a", "out_b"]

        restored = score_from_dict(data)
        assert restored.tasks[0].outputs == ("out_a", "out_b")


class TestDependencyKind:
    """Tests for DependencyKind enum (v0.13 Phase 2 A1-A2)."""

    def test_enum_values(self):
        from sonata.score import DependencyKind
        assert DependencyKind.DATA == "data"
        assert DependencyKind.STORAGE == "storage"
        assert DependencyKind.WAR == "war"
        assert DependencyKind.ORDERING == "ordering"

    def test_from_str(self):
        from sonata.score import DependencyKind
        assert DependencyKind.from_str("data") == DependencyKind.DATA
        assert DependencyKind.from_str("STORAGE") == DependencyKind.STORAGE
        assert DependencyKind.from_str("unknown") == DependencyKind.DATA  # default

    def test_dependency_default_kind(self):
        from sonata.score import Dependency, DependencyKind
        d = Dependency(producer=0, consumer=1)
        assert d.kind == DependencyKind.DATA

    def test_dependency_with_explicit_kind(self):
        from sonata.score import Dependency, DependencyKind
        d = Dependency(producer=0, consumer=1, kind=DependencyKind.WAR)
        assert d.kind == DependencyKind.WAR
        assert d.kind.value == "war"

    def test_dependency_kind_serialization(self):
        """kind serializes as string value, not enum repr."""
        from sonata.score import Dependency, DependencyKind, Score, RuntimeTarget
        from sonata.serialization import score_to_dict
        rt = RuntimeTarget(runtime="host_build_graph", function_name="f", aicpu_thread_num=1)
        dep = Dependency(producer=0, consumer=1, kind=DependencyKind.STORAGE)
        score = Score(name="test", runtime_target=rt, tasks=(),
                      dependencies=(dep,), shape_assumptions=())
        data = score_to_dict(score)
        assert data["dependencies"][0]["kind"] == "storage"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
