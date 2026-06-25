# Copyright (c) PyPTO Contributors.
# ---------------------------------------------------------------------------
"""v0.28 Phase A2: dispatch_regions output as first-class assertions.

Verifies that ``dispatch_regions()`` produces correct counts for:
- All-static, all-dynamic, and mixed region configurations.
- Edge cases: empty region list, single region, ineligible result.
"""

from sonata.pipeline import SonataAnalysisResult, dispatch_regions


class TestPhaseA2DispatchAssertions:
    """dispatch_regions() output is predictable and assertable."""

    # ── All-same type ──

    def test_all_static(self):
        """All static → dispatch shows 0 fallback."""
        result = SonataAnalysisResult(
            eligible=True,
            region_statuses={"r0": "static", "r1": "static", "r2": "static"},
        )
        d = dispatch_regions(result)
        assert d.optimized_count == 3
        assert d.fallback_count == 0

    def test_all_dynamic(self):
        """All dynamic → dispatch shows 0 optimized."""
        result = SonataAnalysisResult(
            eligible=True,
            region_statuses={"r0": "dynamic", "r1": "dynamic"},
        )
        d = dispatch_regions(result)
        assert d.optimized_count == 0
        assert d.fallback_count == 2

    # ── Mixed ──

    def test_mixed_counts(self):
        """Mixed static+dynamic → dispatch counts sum correctly."""
        result = SonataAnalysisResult(
            eligible=True,
            region_statuses={"static": "static", "dynamic": "dynamic"},
        )
        d = dispatch_regions(result)
        assert d.optimized_count == 1
        assert d.fallback_count == 1

    def test_mixed_many_regions(self):
        """Many regions → dispatch counts match."""
        statuses = {f"s{i}": "static" for i in range(5)}
        statuses.update({f"d{i}": "dynamic" for i in range(3)})
        result = SonataAnalysisResult(eligible=True, region_statuses=statuses)
        d = dispatch_regions(result)
        assert d.optimized_count == 5
        assert d.fallback_count == 3

    # ── Edge cases ──

    def test_single_static(self):
        """Single static region → 1 optimized, 0 fallback."""
        result = SonataAnalysisResult(
            eligible=True,
            region_statuses={"r0": "static"},
        )
        d = dispatch_regions(result)
        assert d.optimized_count == 1
        assert d.fallback_count == 0

    def test_single_dynamic(self):
        """Single dynamic region → 0 optimized, 1 fallback."""
        result = SonataAnalysisResult(
            eligible=True,
            region_statuses={"r0": "dynamic"},
        )
        d = dispatch_regions(result)
        assert d.optimized_count == 0
        assert d.fallback_count == 1

    def test_empty_regions(self):
        """Empty region list → 0 optimized, 0 fallback."""
        result = SonataAnalysisResult(
            eligible=True,
            region_statuses={},
        )
        d = dispatch_regions(result)
        assert d.optimized_count == 0
        assert d.fallback_count == 0

    def test_ineligible_empty_dispatch(self):
        """Ineligible result (eligible=False) → dispatch still returns 0/0."""
        result = SonataAnalysisResult(eligible=False)
        d = dispatch_regions(result)
        assert d.optimized_count == 0
        assert d.fallback_count == 0
