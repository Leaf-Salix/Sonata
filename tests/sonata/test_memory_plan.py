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


def _ranges_overlap(a: BufferAllocation, b: BufferAllocation) -> bool:
    return a.offset < b.end and b.offset < a.end
