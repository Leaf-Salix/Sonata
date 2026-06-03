# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Performance benchmarks for v0.17 optimizations.

Measures eligibility + region analysis time at qwen3-scale (49 tasks)
and verifies IR tree walk cache effectiveness.

Usage:
    PYTHONPATH=src python -m pytest tests/sonata/test_performance.py -v
"""

import time
from types import SimpleNamespace

from sonata.eligibility import _walk, _walk_cache, check_static_eligibility
from sonata.regions import extract_regions, build_region_tree, check_region_eligibility


def _make_call(idx: int):
    """Create a mock Call node with arg_directions."""
    return SimpleNamespace(
        __class__=type("Call", (), {}),
        callee_name=f"kernel_{idx}",
        args=("x", "y"),
        arg_names=("x", "y"),
        arg_directions=("Input", "Output"),
        arg_storage_keys=("param:x", None),
        core_type="aic",
        node=SimpleNamespace(),
    )


def _make_function(n_calls: int, name: str = "orch"):
    """Create a mock Orchestration function with n_calls ordinary calls."""
    calls = [_make_call(i) for i in range(n_calls)]
    body = [SimpleNamespace(__class__=type("Call", (), {})) for _ in range(n_calls)]
    return SimpleNamespace(
        body=body,
        name=name,
        calls=calls,
        func_type=SimpleNamespace(value="Orchestration"),
        node=SimpleNamespace(body=body, name=name),
    )


def _make_program(functions: list):
    """Create a mock program with given functions."""
    return SimpleNamespace(
        functions={f.name: f for f in functions},
        body=[f.node for f in functions],
    )


class TestSonataAnalyzePerformance:
    """v0.17 C3: Benchmark sonata_analyze at qwen3 scale."""

    def test_region_analysis_qwen3_scale(self):
        """Region analysis at qwen3 scale (49 calls) completes in <2s."""
        func = _make_function(49, name="qwen3_decode")
        node = func.node

        start = time.monotonic()
        region_map = extract_regions(node)
        tree = build_region_tree(region_map)
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, f"qwen3-scale region analysis took {elapsed:.3f}s (limit 2s)"

    def test_region_analysis_deepseek_scale(self):
        """Region analysis at deepseek scale (46 calls) completes in <2s."""
        func = _make_function(46, name="deepseek_decode")
        node = func.node

        start = time.monotonic()
        region_map = extract_regions(node)
        tree = build_region_tree(region_map)
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, f"deepseek-scale region analysis took {elapsed:.3f}s (limit 2s)"


class TestWalkCachePerformance:
    """v0.17 C2: Verify walk cache improves repeated traversal."""

    def test_walk_cache_speedup(self):
        """Cached walk is >= 2x faster than uncached walk."""
        func = _make_function(100, name="bench")
        node = func.node

        # Clear cache
        _walk_cache.pop(id(node), None)

        # Uncached walk
        start = time.monotonic()
        r1 = tuple(_walk(node))
        uncached_time = time.monotonic() - start

        # Cached walk (should be faster)
        start = time.monotonic()
        r2 = tuple(_walk(node))
        cached_time = time.monotonic() - start

        assert r1 == r2, "Cached walk should return same results"
        assert len(r1) > 0

        # Cached should be at least 2x faster (usually much more)
        if uncached_time > 0.001:  # Only check if uncached took measurable time
            speedup = uncached_time / cached_time
            assert speedup >= 2.0, (
                f"Walk cache speedup {speedup:.1f}x (uncached={uncached_time:.4f}s, "
                f"cached={cached_time:.4f}s)"
            )

        # Cleanup
        _walk_cache.pop(id(node), None)


class TestLargeGraphPerformance:
    """v0.19 Phase 2 B2: Large graph (100+ tasks) performance."""

    def test_region_analysis_200_calls(self):
        """Region analysis at 200 calls completes in <5s (no O(N²))."""
        func = _make_function(200, name="large_graph")
        node = func.node

        start = time.monotonic()
        region_map = extract_regions(node)
        tree = build_region_tree(region_map)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"200-call graph took {elapsed:.3f}s (limit 5s)"

    def test_region_analysis_scales_linearly(self):
        """Region extraction time scales linearly, not quadratically."""
        times = []
        for n in (50, 100, 200):
            func = _make_function(n, name=f"graph_{n}")
            start = time.monotonic()
            region_map = extract_regions(func.node)
            elapsed = time.monotonic() - start
            times.append((n, elapsed))

        # 4x size should not take >10x time (linear, not quadratic)
        if times[0][1] > 0.001:
            ratio = times[2][1] / times[0][1]
            assert ratio < 10.0, (
                f"Not linear: 4x size took {ratio:.1f}x time "
                f"({times[0][1]:.4f}s → {times[2][1]:.4f}s)"
            )

    def test_walk_cache_200_calls(self):
        """Walk cache works correctly for 200-call graph."""
        func = _make_function(200, name="large_graph")
        node = func.node

        _walk_cache.pop(id(node), None)

        r1 = tuple(_walk(node))
        r2 = tuple(_walk(node))

        assert r1 == r2
        assert len(r1) > 200

        _walk_cache.pop(id(node), None)
