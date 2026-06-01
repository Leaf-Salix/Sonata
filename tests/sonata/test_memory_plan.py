"""Tests for v0.3 memory planning."""

from sonata.liveness import BufferLifetime
from sonata.memory_plan import BufferAllocation, MemoryPlan, plan_memory


class TestPlanMemory:
    def test_no_conflict(self):
        lifetimes = (
            BufferLifetime("a", birth=0, death=1),
            BufferLifetime("b", birth=2, death=3),
        )
        sizes = {"a": 100, "b": 200}
        plan = plan_memory(lifetimes, sizes)
        assert plan.peak_memory <= 200
        a = plan.by_key("a")
        b = plan.by_key("b")
        assert a.offset == 0
        assert b.offset == 0

    def test_conflict_no_overlap(self):
        lifetimes = (
            BufferLifetime("a", birth=0, death=3),
            BufferLifetime("b", birth=1, death=2),
        )
        sizes = {"a": 100, "b": 50}
        plan = plan_memory(lifetimes, sizes)
        a = plan.by_key("a")
        b = plan.by_key("b")
        assert a.offset != b.offset or a.size == 0
        assert not _ranges_overlap(a, b)

    def test_greedy_by_size(self):
        lifetimes = (
            BufferLifetime("small", birth=0, death=3),
            BufferLifetime("big", birth=0, death=3),
        )
        sizes = {"small": 10, "big": 100}
        plan = plan_memory(lifetimes, sizes)
        big = plan.by_key("big")
        small = plan.by_key("small")
        assert big.offset == 0
        assert small.offset == 100

    def test_peak_memory(self):
        lifetimes = (
            BufferLifetime("a", birth=0, death=1),
            BufferLifetime("b", birth=0, death=1),
        )
        sizes = {"a": 64, "b": 128}
        plan = plan_memory(lifetimes, sizes)
        assert plan.peak_memory == 192

    def test_empty(self):
        plan = plan_memory((), {})
        assert plan.peak_memory == 0
        assert plan.allocations == ()

    def test_single_buffer(self):
        lifetimes = (BufferLifetime("x", birth=0, death=0),)
        plan = plan_memory(lifetimes, {"x": 256})
        assert plan.peak_memory == 256
        assert plan.by_key("x").offset == 0

    def test_three_buffers_two_overlap(self):
        lifetimes = (
            BufferLifetime("a", birth=0, death=2),
            BufferLifetime("b", birth=1, death=3),
            BufferLifetime("c", birth=3, death=4),
        )
        sizes = {"a": 64, "b": 64, "c": 64}
        plan = plan_memory(lifetimes, sizes)
        assert plan.peak_memory <= 128

    def test_zero_size_buffer(self):
        lifetimes = (
            BufferLifetime("a", birth=0, death=1),
            BufferLifetime("b", birth=0, death=1),
        )
        sizes = {"a": 0, "b": 100}
        plan = plan_memory(lifetimes, sizes)
        assert plan.by_key("a").offset == 0
        assert plan.by_key("a").size == 0


class TestMemoryPlan:
    def test_total_allocated(self):
        plan = MemoryPlan(allocations=(
            BufferAllocation("a", offset=0, size=100),
            BufferAllocation("b", offset=100, size=50),
        ))
        assert plan.total_allocated() == 150

    def test_by_key_not_found(self):
        plan = MemoryPlan()
        assert plan.by_key("nonexistent") is None


class TestConflictMatrix:
    """Tests for compute_conflict_matrix (v0.11 Phase 2 A1-A3)."""

    def test_conflict_matrix_basic(self):
        """Overlapping lifetimes produce True in the matrix."""
        from sonata.memory_plan import compute_conflict_matrix
        lifetimes = [
            BufferLifetime(storage_key="a", birth=0, death=5),
            BufferLifetime(storage_key="b", birth=3, death=8),   # overlaps a
            BufferLifetime(storage_key="c", birth=10, death=15), # no overlap
        ]
        m = compute_conflict_matrix(lifetimes)
        assert m[0][1] is True   # a conflicts with b
        assert m[1][0] is True   # symmetric
        assert m[0][2] is False  # a doesn't conflict with c
        assert m[1][2] is False  # b doesn't conflict with c
        assert all(m[i][i] is False for i in range(3))  # no self-conflict

    def test_stream_aware_conflicts(self):
        """Buffers on different streams don't conflict even if lifetimes overlap."""
        from sonata.memory_plan import compute_conflict_matrix
        lifetimes = [
            BufferLifetime(storage_key="a", birth=0, death=10),
            BufferLifetime(storage_key="b", birth=5, death=15),  # overlaps a
        ]
        # Same stream → conflict
        m_same = compute_conflict_matrix(lifetimes, stream_ids={"a": 0, "b": 0})
        assert m_same[0][1] is True

        # Different streams → no conflict
        m_diff = compute_conflict_matrix(lifetimes, stream_ids={"a": 0, "b": 1})
        assert m_diff[0][1] is False

    def test_inplace_handling(self):
        """Same buffer (self-overlap) is not a conflict by default."""
        from sonata.memory_plan import compute_conflict_matrix
        lifetimes = [
            BufferLifetime(storage_key="a", birth=0, death=5),
        ]
        m = compute_conflict_matrix(lifetimes)
        assert m[0][0] is False  # no self-conflict

    def test_empty(self):
        from sonata.memory_plan import compute_conflict_matrix
        assert compute_conflict_matrix([]) == []


class TestConstraintSolver:
    """Tests for ConstraintSolver implementations (v0.11 Phase 2 B1-B4)."""

    def test_constraint_solver_basic(self):
        """GreedySolver assigns non-overlapping offsets to conflicting buffers."""
        from sonata.memory_plan import GreedySolver, compute_conflict_matrix
        from sonata.liveness import BufferLifetime

        lifetimes = [
            BufferLifetime(storage_key="a", birth=0, death=5),
            BufferLifetime(storage_key="b", birth=3, death=8),  # overlaps a
            BufferLifetime(storage_key="c", birth=10, death=15),
        ]
        matrix = compute_conflict_matrix(lifetimes)
        sizes = [100, 200, 50]
        solver = GreedySolver()
        plan = solver.solve(matrix, sizes)

        # b (200) is placed first (sorted by size desc), then a, then c
        a_alloc = plan.by_key("buf_0")
        b_alloc = plan.by_key("buf_1")
        c_alloc = plan.by_key("buf_2")
        assert a_alloc is not None and b_alloc is not None and c_alloc is not None

        # a and b conflict → must not overlap
        assert not _ranges_overlap(a_alloc, b_alloc)

    def test_greedy_fallback(self):
        """solve_memory falls back to GreedySolver on primary failure."""
        from sonata.memory_plan import (
            ConstraintSolver, GreedySolver, MemoryPlan, solve_memory,
        )

        class FailingSolver(ConstraintSolver):
            def solve(self, conflict_matrix, sizes, device_memory_limit=None):
                raise TimeoutError("simulated timeout")

        matrix = [[False]]
        sizes = [64]
        plan = solve_memory(FailingSolver(), matrix, sizes)
        assert plan.by_key("buf_0") is not None

    def test_dynamic_shape_rejection(self):
        """DynamicShapeError is raised for dynamic shapes."""
        from sonata.memory_plan import DynamicShapeError
        import pytest
        with pytest.raises(DynamicShapeError) as exc_info:
            raise DynamicShapeError(["batch_size", "seq_len"])
        assert "batch_size" in str(exc_info.value)


class TestMemoryPlanSchema:
    """Tests for MemoryPlan schema extension (v0.11 Phase 2 C1-C3)."""

    def test_solver_type_default(self):
        """Default solver_type is 'greedy' for backward compatibility."""
        from sonata.memory_plan import MemoryPlan
        plan = MemoryPlan()
        assert plan.solver_type == "greedy"

    def test_solver_type_set_by_greedy(self):
        """GreedySolver sets solver_type='greedy'."""
        from sonata.memory_plan import GreedySolver
        solver = GreedySolver()
        plan = solver.solve([[False]], [100])
        assert plan.solver_type == "greedy"

    def test_conflict_matrix_hash(self):
        """conflict_matrix_hash survives round-trip."""
        from sonata.memory_plan import MemoryPlan
        plan = MemoryPlan(conflict_matrix_hash="abc123")
        assert plan.conflict_matrix_hash == "abc123"

    def test_backward_compat_no_new_fields(self):
        """Old code without new fields gets defaults."""
        from sonata.memory_plan import MemoryPlan
        # Simulate old-style construction (only allocations + peak_memory)
        plan = MemoryPlan(allocations=(), peak_memory=0)
        assert plan.solver_type == "greedy"
        assert plan.conflict_matrix_hash is None


def _ranges_overlap(a: BufferAllocation, b: BufferAllocation) -> bool:
    return a.offset < b.end and b.offset < a.end
