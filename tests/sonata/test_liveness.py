"""Tests for v0.3 liveness analysis."""

from sonata.liveness import BufferLifetime, StorageConflict, compute_lifetimes, find_conflicts
from sonata.score import Task


class TestComputeLifetimes:
    def test_single_buffer(self):
        tasks = (
            Task(task_id=0, func_id=0, core_type="aic",
                 args=("x", "y"), arg_directions=("input", "output"),
                 arg_storage_keys=("param:x", "alloc:y")),
        )
        lifetimes = compute_lifetimes(tasks)
        assert len(lifetimes) == 2
        x_lt = next(lt for lt in lifetimes if lt.storage_key == "param:x")
        assert x_lt.birth == 0 and x_lt.death == 0
        y_lt = next(lt for lt in lifetimes if lt.storage_key == "alloc:y")
        assert y_lt.birth == 0 and y_lt.death == 0

    def test_buffer_written_then_read(self):
        tasks = (
            Task(task_id=0, func_id=0, core_type="aic",
                 args=("x", "y"), arg_directions=("input", "output"),
                 arg_storage_keys=("param:x", "alloc:y")),
            Task(task_id=1, func_id=1, core_type="aiv",
                 args=("y", "z"), arg_directions=("input", "output"),
                 arg_storage_keys=("alloc:y", "alloc:z")),
        )
        lifetimes = compute_lifetimes(tasks)
        y_lt = next(lt for lt in lifetimes if lt.storage_key == "alloc:y")
        assert y_lt.birth == 0
        assert y_lt.death == 1

    def test_multiple_reads_extend_death(self):
        tasks = (
            Task(task_id=0, func_id=0, core_type="aic",
                 args=("x", "y"), arg_directions=("input", "output"),
                 arg_storage_keys=("param:x", "alloc:y")),
            Task(task_id=1, func_id=1, core_type="aiv",
                 args=("y",), arg_directions=("input",),
                 arg_storage_keys=("alloc:y",)),
            Task(task_id=2, func_id=1, core_type="aiv",
                 args=("y",), arg_directions=("input",),
                 arg_storage_keys=("alloc:y",)),
        )
        lifetimes = compute_lifetimes(tasks)
        y_lt = next(lt for lt in lifetimes if lt.storage_key == "alloc:y")
        assert y_lt.birth == 0
        assert y_lt.death == 2

    def test_skips_unknown_storage_keys(self):
        tasks = (
            Task(task_id=0, func_id=0, core_type="aic",
                 args=("x", "y"), arg_directions=("input", "output"),
                 arg_storage_keys=(None, "alloc:y")),
        )
        lifetimes = compute_lifetimes(tasks)
        assert len(lifetimes) == 1
        assert lifetimes[0].storage_key == "alloc:y"

    def test_skips_scalar_args(self):
        tasks = (
            Task(task_id=0, func_id=0, core_type="aic",
                 args=("x", "42"), arg_directions=("input", "scalar"),
                 arg_storage_keys=("param:x", None)),
        )
        lifetimes = compute_lifetimes(tasks)
        assert len(lifetimes) == 1

    def test_empty_tasks(self):
        assert compute_lifetimes(()) == ()


class TestBufferLifetime:
    def test_overlaps_true(self):
        a = BufferLifetime("a", birth=0, death=3)
        b = BufferLifetime("b", birth=2, death=5)
        assert a.overlaps(b)
        assert b.overlaps(a)

    def test_overlaps_false(self):
        a = BufferLifetime("a", birth=0, death=2)
        b = BufferLifetime("b", birth=3, death=5)
        assert not a.overlaps(b)

    def test_overlaps_adjacent(self):
        """Touching lifetimes (death==birth) do NOT overlap — PyPTO semantics."""
        a = BufferLifetime("a", birth=0, death=2)
        b = BufferLifetime("b", birth=2, death=4)
        assert not a.overlaps(b)

    def test_overlaps_contained(self):
        a = BufferLifetime("a", birth=0, death=10)
        b = BufferLifetime("b", birth=3, death=5)
        assert a.overlaps(b)


class TestFindConflicts:
    def test_no_conflicts(self):
        lifetimes = (
            BufferLifetime("a", birth=0, death=1),
            BufferLifetime("b", birth=2, death=3),
        )
        assert find_conflicts(lifetimes) == ()

    def test_one_conflict(self):
        lifetimes = (
            BufferLifetime("a", birth=0, death=3),
            BufferLifetime("b", birth=2, death=5),
        )
        conflicts = find_conflicts(lifetimes)
        assert len(conflicts) == 1
        assert conflicts[0].key_a == "a"
        assert conflicts[0].key_b == "b"

    def test_multiple_conflicts(self):
        lifetimes = (
            BufferLifetime("a", birth=0, death=5),
            BufferLifetime("b", birth=1, death=3),
            BufferLifetime("c", birth=2, death=4),
        )
        conflicts = find_conflicts(lifetimes)
        assert len(conflicts) == 3

    def test_empty(self):
        assert find_conflicts(()) == ()


class TestMultiOutputLiveness:
    """Tests for compute_lifetimes with Task.outputs (v0.11 Phase 3 B1-B3)."""

    def test_multi_output_liveness_accuracy(self):
        """Explicit outputs produce accurate lifetimes."""
        from sonata.score import Task
        tasks = (
            Task(task_id=0, func_id=1, core_type="aicore",
                 outputs=("out_a", "out_b")),
            Task(task_id=1, func_id=2, core_type="aicore",
                 args=("out_a",), arg_directions=("Input",),
                 arg_storage_keys=("out_a",)),
        )
        lifetimes = compute_lifetimes(tasks)
        by_key = {lt.storage_key: lt for lt in lifetimes}

        assert "out_a" in by_key
        assert "out_b" in by_key
        assert by_key["out_a"].birth == 0
        assert by_key["out_a"].death == 1  # read by task 1
        assert by_key["out_b"].birth == 0
        assert by_key["out_b"].death == 0  # only written, never read

    def test_outputs_no_double_count(self):
        """Output buffers already in arg_directions aren't duplicated."""
        from sonata.score import Task
        tasks = (
            Task(task_id=0, func_id=1, core_type="aicore",
                 args=("x",), arg_directions=("Output",),
                 arg_storage_keys=("buf_x",),
                 outputs=("buf_x",)),
        )
        lifetimes = compute_lifetimes(tasks)
        keys = [lt.storage_key for lt in lifetimes]
        assert keys.count("buf_x") == 1


class TestOutputDefsLiveness:
    """v0.21 Phase 4 A2: compute_lifetimes uses output_defs when available."""

    def test_output_defs_tracked_as_writes(self):
        """Task with output_defs → buffers tracked as writes."""
        from sonata.liveness import compute_lifetimes
        from sonata.score import Task, OutputDef
        tasks = (
            Task(task_id=0, func_id=1, core_type="aic",
                 output_defs=(
                     OutputDef(buffer_id="out_a", dtype="fp16"),
                     OutputDef(buffer_id="out_b", dtype="fp32"),
                 )),
        )
        lifetimes = compute_lifetimes(tasks)
        by_key = {lt.storage_key: lt for lt in lifetimes}
        assert "out_a" in by_key
        assert "out_b" in by_key
        assert by_key["out_a"].birth == 0
        assert by_key["out_a"].death == 0

    def test_output_defs_with_reads(self):
        """output_defs outputs + arg reads → correct lifetime."""
        from sonata.liveness import compute_lifetimes
        from sonata.score import Task, OutputDef
        tasks = (
            Task(task_id=0, func_id=1, core_type="aic",
                 output_defs=(OutputDef(buffer_id="out_a"),)),
            Task(task_id=1, func_id=2, core_type="aic",
                 args=("out_a",), arg_directions=("Input",),
                 arg_storage_keys=("out_a",)),
        )
        lifetimes = compute_lifetimes(tasks)
        by_key = {lt.storage_key: lt for lt in lifetimes}
        assert by_key["out_a"].birth == 0
        assert by_key["out_a"].death == 1  # read by task 1

    def test_output_defs_none_fallback(self):
        """Task with output_defs=None → falls back to outputs/arg_directions."""
        from sonata.liveness import compute_lifetimes
        from sonata.score import Task
        tasks = (
            Task(task_id=0, func_id=1, core_type="aic",
                 outputs=("buf_x",), output_defs=None),
        )
        lifetimes = compute_lifetimes(tasks)
        by_key = {lt.storage_key: lt for lt in lifetimes}
        assert "buf_x" in by_key

    def test_output_defs_empty_tuple(self):
        """Task with output_defs=() → no extra outputs tracked."""
        from sonata.liveness import compute_lifetimes
        from sonata.score import Task
        tasks = (
            Task(task_id=0, func_id=1, core_type="aic",
                 output_defs=()),
        )
        lifetimes = compute_lifetimes(tasks)
        assert len(lifetimes) == 0
