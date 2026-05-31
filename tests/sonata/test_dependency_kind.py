"""Tests for v0.3 dependency kind classification and ordering dependencies."""

from sonata.dependencies import (
    build_dataflow_dependencies,
    build_mixed_dependencies,
    build_ordering_dependencies,
)
from sonata.score import Dependency, Task


class TestDependencyKind:
    def test_raw_edge_is_data(self):
        tasks = (
            Task(task_id=0, func_id=0, core_type="aic",
                 args=("x", "y"), arg_directions=("input", "output"),
                 arg_storage_keys=("buf:x", "buf:y")),
            Task(task_id=1, func_id=1, core_type="aiv",
                 args=("y", "z"), arg_directions=("input", "output"),
                 arg_storage_keys=("buf:y", "buf:z")),
        )
        deps = build_dataflow_dependencies(tasks)
        raw = [d for d in deps if d.producer == 0 and d.consumer == 1]
        assert len(raw) == 1
        assert raw[0].kind == "data"

    def test_waw_edge_is_storage(self):
        tasks = (
            Task(task_id=0, func_id=0, core_type="aic",
                 args=("x", "y"), arg_directions=("input", "output"),
                 arg_storage_keys=("buf:x", "buf:y")),
            Task(task_id=1, func_id=1, core_type="aiv",
                 args=("x", "y"), arg_directions=("input", "output"),
                 arg_storage_keys=("buf:x", "buf:y")),
        )
        deps = build_dataflow_dependencies(tasks)
        waw = [d for d in deps if d.kind == "storage"]
        assert len(waw) >= 1

    def test_war_edge(self):
        tasks = (
            Task(task_id=0, func_id=0, core_type="aic",
                 args=("x", "y"), arg_directions=("input", "output"),
                 arg_storage_keys=("buf:x", "buf:y")),
            Task(task_id=1, func_id=1, core_type="aiv",
                 args=("y",), arg_directions=("input",),
                 arg_storage_keys=("buf:y",)),
            Task(task_id=2, func_id=2, core_type="aic",
                 args=("y",), arg_directions=("inout",),
                 arg_storage_keys=("buf:y",)),
        )
        deps = build_dataflow_dependencies(tasks)
        war = [d for d in deps if d.kind == "war"]
        assert len(war) >= 1

    def test_default_kind_is_data(self):
        dep = Dependency(producer=0, consumer=1)
        assert dep.kind == "data"

    def test_kind_preserved(self):
        dep = Dependency(producer=0, consumer=1, kind="ordering")
        assert dep.kind == "ordering"


class TestOrderingDependencies:
    def test_chain_all(self):
        tasks = (
            Task(task_id=0, func_id=0, core_type="aic"),
            Task(task_id=1, func_id=1, core_type="aiv"),
            Task(task_id=2, func_id=2, core_type="aic"),
        )
        deps = build_ordering_dependencies(tasks)
        assert len(deps) == 2
        assert all(d.kind == "ordering" for d in deps)

    def test_side_effect_filter(self):
        tasks = (
            Task(task_id=0, func_id=0, core_type="aic"),
            Task(task_id=1, func_id=1, core_type="aiv"),
            Task(task_id=2, func_id=2, core_type="aic"),
        )
        deps = build_ordering_dependencies(
            tasks, side_effect_tasks=frozenset({0, 2}),
        )
        assert len(deps) == 1
        assert deps[0].producer == 0
        assert deps[0].consumer == 2

    def test_empty_tasks(self):
        assert build_ordering_dependencies(()) == ()


class TestMixedDependencies:
    def test_dataflow_plus_ordering(self):
        tasks = (
            Task(task_id=0, func_id=0, core_type="aic",
                 args=("x", "y"), arg_directions=("input", "output"),
                 arg_storage_keys=("buf:x", "buf:y")),
            Task(task_id=1, func_id=1, core_type="aiv",
                 args=("y", "z"), arg_directions=("input", "output"),
                 arg_storage_keys=("buf:y", "buf:z")),
            Task(task_id=2, func_id=2, core_type="aic",
                 args=("w",), arg_directions=("input",),
                 arg_storage_keys=("buf:w",)),
        )
        deps = build_mixed_dependencies(
            tasks, side_effect_tasks=frozenset({0, 2}),
        )
        data_deps = [d for d in deps if d.kind == "data"]
        ordering_deps = [d for d in deps if d.kind == "ordering"]
        assert len(data_deps) >= 1
        assert len(ordering_deps) >= 1

    def test_no_duplicate_ordering_when_dataflow_exists(self):
        tasks = (
            Task(task_id=0, func_id=0, core_type="aic",
                 args=("x", "y"), arg_directions=("input", "output"),
                 arg_storage_keys=("buf:x", "buf:y")),
            Task(task_id=1, func_id=1, core_type="aiv",
                 args=("y", "z"), arg_directions=("input", "output"),
                 arg_storage_keys=("buf:y", "buf:z")),
        )
        deps = build_mixed_dependencies(
            tasks, side_effect_tasks=frozenset({0, 1}),
        )
        pairs = [(d.producer, d.consumer) for d in deps]
        assert len(pairs) == len(set(pairs))
