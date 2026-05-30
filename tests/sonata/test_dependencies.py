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
    build_dataflow_dependencies,
    build_dependencies,
    build_sequential_dependencies,
    supports_dataflow_dependencies,
)
from sonata.dependencies import dataflow_dependency_fallback_code
from sonata.score import Task


def test_sequential_dependencies_chain_every_neighbor() -> None:
    tasks = (
        Task(task_id=0, func_id=0, core_type="aiv"),
        Task(task_id=1, func_id=1, core_type="aiv"),
        Task(task_id=2, func_id=2, core_type="aiv"),
    )

    deps = build_sequential_dependencies(tasks)

    assert [(dep.producer, dep.consumer) for dep in deps] == [(0, 1), (1, 2)]


def test_dataflow_dependencies_skip_independent_readers() -> None:
    tasks = (
        Task(task_id=0, func_id=0, core_type="aiv", args=("x",), arg_directions=("Input",)),
        Task(task_id=1, func_id=1, core_type="aiv", args=("y",), arg_directions=("Input",)),
        Task(
            task_id=2,
            func_id=2,
            core_type="aiv",
            args=("x", "out"),
            arg_directions=("Input", "OutputExisting"),
        ),
    )

    deps = build_dataflow_dependencies(tasks)

    assert deps == ()


def test_dataflow_dependencies_add_raw_edge() -> None:
    tasks = (
        Task(
            task_id=0,
            func_id=0,
            core_type="aiv",
            args=("x", "tmp"),
            arg_directions=("Input", "OutputExisting"),
        ),
        Task(task_id=1, func_id=1, core_type="aiv", args=("tmp",), arg_directions=("Input",)),
    )

    deps = build_dataflow_dependencies(tasks)

    assert [(dep.producer, dep.consumer) for dep in deps] == [(0, 1)]


def test_dataflow_dependencies_prefer_storage_keys_over_arg_names() -> None:
    tasks = (
        Task(
            task_id=0,
            func_id=0,
            core_type="aiv",
            args=("tmp_v1",),
            arg_directions=("OutputExisting",),
            arg_storage_keys=("buffer:tmp",),
        ),
        Task(
            task_id=1,
            func_id=1,
            core_type="aiv",
            args=("tmp_v2",),
            arg_directions=("Input",),
            arg_storage_keys=("buffer:tmp",),
        ),
    )

    deps = build_dataflow_dependencies(tasks)

    assert [(dep.producer, dep.consumer) for dep in deps] == [(0, 1)]


def test_dataflow_dependencies_treat_distinct_storage_keys_as_independent() -> None:
    tasks = (
        Task(
            task_id=0,
            func_id=0,
            core_type="aiv",
            args=("tmp",),
            arg_directions=("OutputExisting",),
            arg_storage_keys=("buffer:a",),
        ),
        Task(
            task_id=1,
            func_id=1,
            core_type="aiv",
            args=("tmp",),
            arg_directions=("Input",),
            arg_storage_keys=("buffer:b",),
        ),
    )

    deps = build_dataflow_dependencies(tasks)

    assert deps == ()


def test_dataflow_dependencies_add_waw_edge() -> None:
    tasks = (
        Task(task_id=0, func_id=0, core_type="aiv", args=("tmp",), arg_directions=("OutputExisting",)),
        Task(task_id=1, func_id=1, core_type="aiv", args=("tmp",), arg_directions=("InOut",)),
    )

    deps = build_dataflow_dependencies(tasks)

    assert [(dep.producer, dep.consumer) for dep in deps] == [(0, 1)]


def test_dataflow_dependencies_add_war_edge() -> None:
    tasks = (
        Task(task_id=0, func_id=0, core_type="aiv", args=("tmp",), arg_directions=("Input",)),
        Task(task_id=1, func_id=1, core_type="aiv", args=("tmp",), arg_directions=("OutputExisting",)),
    )

    deps = build_dataflow_dependencies(tasks)

    assert [(dep.producer, dep.consumer) for dep in deps] == [(0, 1)]


def test_dataflow_dependencies_ignore_scalar_and_nodep_args() -> None:
    tasks = (
        Task(task_id=0, func_id=0, core_type="aiv", args=("tmp",), arg_directions=("NoDep",)),
        Task(task_id=1, func_id=1, core_type="aiv", args=("tmp",), arg_directions=("Scalar",)),
    )

    deps = build_dataflow_dependencies(tasks)

    assert deps == ()


def test_dataflow_dependencies_require_complete_directions() -> None:
    tasks = (Task(task_id=0, func_id=0, core_type="aiv", args=("x",)),)

    with pytest.raises(ValueError, match="dataflow_v0 requires complete task arg_directions"):
        build_dataflow_dependencies(tasks)

    assert supports_dataflow_dependencies(tasks) is False


class TestDataflowFallbackCodes:
    def test_all_complete_returns_none(self) -> None:
        tasks = (
            Task(task_id=0, func_id=0, core_type="aiv", args=("a",), arg_directions=("input",)),
            Task(task_id=1, func_id=1, core_type="aiv", args=("b",), arg_directions=("output",)),
        )
        assert supports_dataflow_dependencies(tasks) is True
        assert dataflow_dependency_fallback_code(tasks) is None

    def test_none_have_directions_returns_unavailable(self) -> None:
        from sonata.fallback import FallbackCode

        tasks = (
            Task(task_id=0, func_id=0, core_type="aiv", args=("a",)),
            Task(task_id=1, func_id=1, core_type="aiv", args=("b",)),
        )
        assert supports_dataflow_dependencies(tasks) is False
        assert dataflow_dependency_fallback_code(tasks) == FallbackCode.DATAFLOW_DIRECTIONS_UNAVAILABLE

    def test_partial_directions_returns_incomplete(self) -> None:
        from sonata.fallback import FallbackCode

        tasks = (
            Task(task_id=0, func_id=0, core_type="aiv", args=("a",), arg_directions=("input",)),
            Task(task_id=1, func_id=1, core_type="aiv", args=("b",)),
        )
        assert supports_dataflow_dependencies(tasks) is False
        assert dataflow_dependency_fallback_code(tasks) == FallbackCode.DATAFLOW_DIRECTIONS_INCOMPLETE

    def test_empty_tasks_returns_none(self) -> None:
        assert supports_dataflow_dependencies(()) is True
        assert dataflow_dependency_fallback_code(()) is None

    def test_mismatched_direction_lengths_returns_incomplete(self) -> None:
        from sonata.fallback import FallbackCode

        tasks = (
            Task(task_id=0, func_id=0, core_type="aiv", args=("a", "b"), arg_directions=("input",)),
        )
        assert supports_dataflow_dependencies(tasks) is False
        assert dataflow_dependency_fallback_code(tasks) == FallbackCode.DATAFLOW_DIRECTIONS_INCOMPLETE

    def test_supports_truthiness_does_not_enter_dataflow_build_for_missing_directions(self) -> None:
        tasks = (Task(task_id=0, func_id=0, core_type="aiv", args=("a",)),)

        if supports_dataflow_dependencies(tasks):
            build_dataflow_dependencies(tasks)


def test_build_dependencies_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="unsupported Sonata dependency policy"):
        build_dependencies((), policy="unknown")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
