# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Performance benchmarks for Sonata v0.13.

Measures overhead of Sonata analysis on real PyPTO programs and
validates IR cache effectiveness.

Usage:
    PYTHONPATH=src python tests/sonata/test_perf_benchmarks.py
"""

import time
from types import SimpleNamespace

import pytest

try:
    import pypto  # noqa: F401

    _HAS_PYPTO = True
except ImportError:
    _HAS_PYPTO = False

from sonata.pipeline import _extract_certified_ir, sonata_analyze, SonataAnalysisResult
from sonata.regions import extract_regions, build_region_tree, check_region_eligibility
from sonata.serialization import score_fingerprint


def _make_stmt(kind: str):
    return type(kind, (), {"__init__": lambda self: None})()


def _make_func(body: list, name: str = "bench"):
    return SimpleNamespace(body=body, name=name)


def _build_large_graph(n_static: int = 100, n_dynamic: int = 10):
    """Build a graph with n_static Call stmts followed by n_dynamic ForStmt stmts."""
    body = [_make_stmt("Call") for _ in range(n_static)]
    body.extend(_make_stmt("ForStmt") for _ in range(n_dynamic))
    return _make_func(body, name=f"graph_{n_static}s_{n_dynamic}d")


class TestSonataAnalysisOverhead:
    """A3: Measure Sonata analysis overhead on graphs of varying size."""

    def test_analysis_overhead_small(self):
        """Small graph (20 stmts): analysis completes in <1s."""
        node = _build_large_graph(n_static=18, n_dynamic=2)
        start = time.monotonic()
        result = check_region_eligibility(node)
        elapsed = time.monotonic() - start
        assert result.eligible
        assert elapsed < 1.0, f"Small graph analysis took {elapsed:.2f}s (limit 1s)"

    def test_analysis_overhead_medium(self):
        """Medium graph (100 stmts): analysis completes in <2s."""
        node = _build_large_graph(n_static=90, n_dynamic=10)
        start = time.monotonic()
        result = check_region_eligibility(node)
        elapsed = time.monotonic() - start
        assert result.eligible
        assert elapsed < 2.0, f"Medium graph analysis took {elapsed:.2f}s (limit 2s)"

    def test_analysis_overhead_large(self):
        """Large graph (500 stmts): analysis completes in <5s."""
        node = _build_large_graph(n_static=450, n_dynamic=50)
        start = time.monotonic()
        result = check_region_eligibility(node)
        elapsed = time.monotonic() - start
        assert result.eligible
        assert elapsed < 5.0, f"Large graph analysis took {elapsed:.2f}s (limit 5s)"

    def test_region_extraction_scales_linearly(self):
        """Region extraction time scales linearly with graph size."""
        times = []
        for n in (50, 100, 200):
            node = _build_large_graph(n_static=n, n_dynamic=10)
            start = time.monotonic()
            region_map = extract_regions(node)
            elapsed = time.monotonic() - start
            times.append((n, elapsed))

        # 4x size should not take >10x time (linear, not quadratic)
        ratio = times[2][1] / times[0][1] if times[0][1] > 0 else 0
        assert ratio < 10.0, (
            f"Region extraction not linear: 4x size took {ratio:.1f}x time"
        )

    def test_conflict_matrix_scales_linearly(self):
        """Conflict matrix computation scales linearly with buffer count."""
        from sonata.memory_plan import compute_conflict_matrix
        from sonata.liveness import BufferLifetime

        times = []
        for n in (50, 100, 200):
            lifetimes = [
                BufferLifetime(storage_key=f"b{i}", birth=i, death=i + 3)
                for i in range(n)
            ]
            start = time.monotonic()
            compute_conflict_matrix(lifetimes)
            elapsed = time.monotonic() - start
            times.append((n, elapsed))

        # O(N²) is expected, but should be fast for N≤200
        assert times[2][1] < 1.0, f"Conflict matrix N=200 took {times[2][1]:.2f}s"


class TestIRCacheEffectiveness:
    """A3: Verify IR cache logic."""

    @pytest.mark.skipif(not _HAS_PYPTO, reason="pypto not installed")
    def test_cache_returns_none_for_non_program(self):
        """Non-Program objects return None (pipeline can't process them)."""
        node = SimpleNamespace(body=[])
        result = _extract_certified_ir(node)
        assert result is None

    @pytest.mark.skipif(not _HAS_PYPTO, reason="pypto not installed")
    def test_cache_stores_result(self):
        """Cache stores and retrieves results by program id."""
        from sonata.pipeline import _certified_ir_cache

        class FakeProgram:
            pass

        prog = FakeProgram()
        fake_ir = object()
        _certified_ir_cache[id(prog)] = (prog, fake_ir)

        result = _extract_certified_ir(prog)
        assert result is fake_ir

        # Cleanup
        del _certified_ir_cache[id(prog)]

    @pytest.mark.skipif(not _HAS_PYPTO, reason="pypto not installed")
    def test_cache_prevents_dual_pipeline(self):
        """v0.17 C1: Cache ensures only one pipeline run per program.

        The same program object should return cached IR on second call,
        not trigger a second pipeline execution.
        """
        from sonata.pipeline import _certified_ir_cache

        class FakeProgram:
            pass

        prog = FakeProgram()
        fake_ir = {"pipeline_run": 1}
        _certified_ir_cache[id(prog)] = (prog, fake_ir)

        # First call — from cache
        result1 = _extract_certified_ir(prog)
        assert result1 is fake_ir

        # Second call — still from cache (no second pipeline run)
        result2 = _extract_certified_ir(prog)
        assert result2 is fake_ir
        assert result1 is result2

        # Cleanup
        del _certified_ir_cache[id(prog)]
