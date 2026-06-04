"""Tests for v0.3 alias analysis."""

from sonata.alias import (
    ALIAS_ALIAS, ALIAS_DISJOINT, ALIAS_INPLACE, ALIAS_VIEW,
    AliasRelation, analyze_aliases,
)


class TestAnalyzeAliases:
    def test_all_disjoint(self):
        keys = ("param:a", "alloc:b", "alloc:c")
        relations = analyze_aliases(keys)
        assert all(r.is_disjoint for r in relations)

    def test_alias_declaration(self):
        keys = ("param:a", "param:a_copy")
        relations = analyze_aliases(
            keys, alias_declarations={"param:a_copy": "param:a"},
        )
        assert len(relations) == 1
        assert relations[0].relation == ALIAS_ALIAS
        assert relations[0].shares_storage

    def test_view_declaration(self):
        keys = ("alloc:full", "alloc:view")
        relations = analyze_aliases(
            keys, view_declarations={"alloc:view": "alloc:full"},
        )
        assert len(relations) == 1
        assert relations[0].relation == ALIAS_VIEW

    def test_inplace_declaration(self):
        keys = ("alloc:a", "alloc:b")
        relations = analyze_aliases(
            keys, inplace_declarations={("alloc:a", "alloc:b")},
        )
        assert len(relations) == 1
        assert relations[0].relation == ALIAS_INPLACE

    def test_inplace_symmetric(self):
        keys = ("alloc:a", "alloc:b")
        relations = analyze_aliases(
            keys, inplace_declarations={("alloc:b", "alloc:a")},
        )
        assert relations[0].relation == ALIAS_INPLACE

    def test_common_view_source(self):
        keys = ("alloc:v1", "alloc:v2", "alloc:src")
        relations = analyze_aliases(
            keys, view_declarations={"alloc:v1": "alloc:src", "alloc:v2": "alloc:src"},
        )
        v1_v2 = next(r for r in relations if r.key_a == "alloc:v1" and r.key_b == "alloc:v2")
        assert v1_v2.relation == ALIAS_VIEW

    def test_single_key(self):
        relations = analyze_aliases(("param:a",))
        assert relations == ()

    def test_empty(self):
        assert analyze_aliases(()) == ()


class TestAliasRelation:
    def test_is_disjoint(self):
        r = AliasRelation("a", "b", ALIAS_DISJOINT)
        assert r.is_disjoint
        assert not r.shares_storage

    def test_shares_storage_alias(self):
        r = AliasRelation("a", "b", ALIAS_ALIAS)
        assert not r.is_disjoint
        assert r.shares_storage

    def test_shares_storage_view(self):
        r = AliasRelation("a", "b", ALIAS_VIEW)
        assert r.shares_storage

    def test_shares_storage_inplace(self):
        r = AliasRelation("a", "b", ALIAS_INPLACE)
        assert r.shares_storage


class TestDeriveAliasesFromTasks:
    """v0.20 Phase 2 A2: derive_aliases_from_tasks tests."""

    def test_inout_is_inplace(self):
        """inout direction with multiple tasks → ALIAS_INPLACE."""
        from sonata.alias import derive_aliases_from_tasks, ALIAS_INPLACE
        from sonata.score import Task
        tasks = (
            Task(task_id=0, func_id=0, core_type="aic",
                 args=("x",), arg_directions=("inout",),
                 arg_storage_keys=("buf:x",)),
            Task(task_id=1, func_id=1, core_type="aiv",
                 args=("x",), arg_directions=("input",),
                 arg_storage_keys=("buf:x",)),
        )
        relations = derive_aliases_from_tasks(tasks)
        inplace = [r for r in relations if r.relation == ALIAS_INPLACE]
        assert len(inplace) >= 1

    def test_output_then_input_is_view(self):
        """output then input on same buffer → ALIAS_VIEW."""
        from sonata.alias import derive_aliases_from_tasks, ALIAS_VIEW
        from sonata.score import Task
        tasks = (
            Task(task_id=0, func_id=0, core_type="aic",
                 args=("x",), arg_directions=("output",),
                 arg_storage_keys=("buf:x",)),
            Task(task_id=1, func_id=1, core_type="aiv",
                 args=("x",), arg_directions=("input",),
                 arg_storage_keys=("buf:x",)),
        )
        relations = derive_aliases_from_tasks(tasks)
        view = [r for r in relations if r.relation == ALIAS_VIEW]
        assert len(view) >= 1

    def test_empty_tasks(self):
        """No tasks → empty relations."""
        from sonata.alias import derive_aliases_from_tasks
        assert derive_aliases_from_tasks(()) == ()

    def test_no_directions_returns_empty(self):
        """Tasks without arg_directions → empty."""
        from sonata.alias import derive_aliases_from_tasks
        from sonata.score import Task
        tasks = (Task(task_id=0, func_id=0, core_type="aic"),)
        assert derive_aliases_from_tasks(tasks) == ()

    def test_different_buffers_no_alias(self):
        """Different buffer_ids → no alias relations."""
        from sonata.alias import derive_aliases_from_tasks
        from sonata.score import Task
        tasks = (
            Task(task_id=0, func_id=0, core_type="aic",
                 args=("x",), arg_directions=("input",),
                 arg_storage_keys=("buf:x",)),
            Task(task_id=1, func_id=1, core_type="aiv",
                 args=("y",), arg_directions=("output",),
                 arg_storage_keys=("buf:y",)),
        )
        relations = derive_aliases_from_tasks(tasks)
        # Different buffers → no relations
        assert len(relations) == 0
