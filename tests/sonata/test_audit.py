# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
from sonata.audit import build_score_metadata, build_task_storage_metadata
from sonata.score import Task


class Entry:
    name = "main"


def test_build_task_storage_metadata_counts_known_and_unknown_storage_keys() -> None:
    tasks = (
        Task(
            task_id=0,
            func_id=0,
            core_type="aiv",
            args=("x", "tmp", 1),
            arg_directions=("Input", "OutputExisting", "Scalar"),
            arg_storage_keys=("param:x", None, None),
            name="kernel",
        ),
    )

    metadata = build_task_storage_metadata(tasks)

    assert metadata["storage_key_coverage"] == {"known": 1, "unknown": 2, "total": 3}
    assert metadata["memory_storage_key_coverage"] == {"known": 1, "unknown": 1, "total": 2}
    assert metadata["unknown_memory_storage_args"] == (
        {
            "task_id": 0,
            "task_name": "kernel",
            "arg_index": 1,
            "arg": "tmp",
            "storage_key": None,
        },
    )


def test_build_task_storage_metadata_records_nodep_without_memory_coverage() -> None:
    tasks = (
        Task(
            task_id=0,
            func_id=0,
            core_type="aiv",
            args=("shared",),
            arg_directions=("NoDep",),
            arg_storage_keys=("param:shared",),
            name="kernel",
        ),
    )

    metadata = build_task_storage_metadata(tasks)

    assert metadata["storage_key_coverage"] == {"known": 1, "unknown": 0, "total": 1}
    assert metadata["memory_storage_key_coverage"] == {"known": 0, "unknown": 0, "total": 0}
    assert metadata["nodep_args"] == (
        {
            "task_id": 0,
            "task_name": "kernel",
            "arg_index": 0,
            "arg": "shared",
            "storage_key": "param:shared",
        },
    )


def test_build_task_storage_metadata_tolerates_mismatched_arity() -> None:
    tasks = (
        Task(
            task_id=0,
            func_id=0,
            core_type="aiv",
            args=("x", "out"),
            arg_directions=("Input",),
            arg_storage_keys=("param:x",),
            name="kernel",
        ),
    )

    metadata = build_task_storage_metadata(tasks)

    assert metadata["storage_key_coverage"] == {"known": 1, "unknown": 1, "total": 2}
    assert metadata["memory_storage_key_coverage"] == {"known": 1, "unknown": 0, "total": 1}


def test_build_score_metadata_records_entry_and_policy_fallback() -> None:
    metadata = build_score_metadata(
        (Entry(),),
        (),
        None,
        "sequential_v0",
        "dataflow_v0",
    )

    assert metadata["entry_name"] == "main"
    assert metadata["dependency_policy"] == "sequential_v0"
    assert metadata["requested_dependency_policy"] == "dataflow_v0"
    assert metadata["dependency_policy_fallback_reason"] == "task arg_directions are incomplete"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
