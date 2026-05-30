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
        shape_assumptions=(ShapeAssumption(symbol="a", dims=(1024,)),),
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
        shape_assumptions=(ShapeAssumption(symbol="x", dims=(16, 32)),),
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
        shape_assumptions=(ShapeAssumption(symbol="", dims=(4, -1)),),
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
            ShapeAssumption(symbol="x", dims=(16,)),
            ShapeAssumption(symbol="x", dims=(32,)),
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
            ShapeAssumption(symbol="x", dims=(-1,)),
            ShapeAssumption(symbol="x", dims=(-1,)),
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
            ShapeAssumption(symbol="", dims=(16,)),
            ShapeAssumption(symbol="", dims=(32,)),
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
        shape_assumptions=(ShapeAssumption(symbol="", dims=("n",)),),
    )

    result = score.validate()

    assert not result.eligible
    assert result.reasons == ("shape assumption symbol must not be empty",)


def test_score_validate_rejects_zero_shape_assumption_dimension() -> None:
    score = Score(
        name="zero_shape_dim",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_invalid_graph"),
        shape_assumptions=(ShapeAssumption(symbol="x", dims=(0, 16)),),
    )

    result = score.validate()

    assert not result.eligible
    assert result.reasons == ("shape assumption x has zero dimension",)


def test_score_validate_rejects_non_integer_shape_assumption_dimension() -> None:
    score = Score(
        name="symbolic_shape_dim",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_invalid_graph"),
        shape_assumptions=(ShapeAssumption(symbol="x", dims=("n", 16)),),
    )

    result = score.validate()

    assert not result.eligible
    assert result.reasons == ("shape assumption x has non-integer dimension",)


def test_score_validate_rejects_bool_shape_assumption_dimension() -> None:
    score = Score(
        name="bool_shape_dim",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_invalid_graph"),
        shape_assumptions=(ShapeAssumption(symbol="x", dims=(True, 16)),),
    )

    result = score.validate()

    assert not result.eligible
    assert result.reasons == ("shape assumption x has non-integer dimension",)


def test_score_validate_reports_each_shape_dim_error_category() -> None:
    score = Score(
        name="multi_bad_shape_dim",
        runtime_target=RuntimeTarget(runtime="host_build_graph", function_name="build_invalid_graph"),
        shape_assumptions=(ShapeAssumption(symbol="x", dims=(0, -1, "n", True)),),
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
