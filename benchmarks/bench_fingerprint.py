#!/usr/bin/env python3
"""Benchmark score_fingerprint computation and cache hit performance.

Measures:
  - Time to compute score_fingerprint() for different Score sizes
  - Cache lookup throughput vs Score rebuild cost (hit rate simulation)

Usage:
    PYTHONPATH=src python benchmarks/bench_fingerprint.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _bench_utils import (
    make_score,
    print_results,
    run_benchmark,
    save_results,
)
from sonata import ScoreCache, cached_score, score_fingerprint


TASK_COUNTS = [1, 10, 50, 100, 500]


def bench_fingerprint_compute(num_tasks: int) -> dict:
    """Benchmark score_fingerprint() computation time."""
    score = make_score(num_tasks, dependency_style="chain")

    def compute():
        return score_fingerprint(score)

    return run_benchmark(
        "fingerprint_compute",
        compute,
        params={"num_tasks": num_tasks},
    )


def bench_fingerprint_with_metadata(num_tasks: int) -> dict:
    """Benchmark fingerprint with include_metadata=True."""
    score = make_score(num_tasks, dependency_style="chain")

    def compute():
        return score_fingerprint(score, include_metadata=True)

    return run_benchmark(
        "fingerprint_with_metadata",
        compute,
        params={"num_tasks": num_tasks},
    )


def bench_cache_hit_vs_miss(num_tasks: int) -> dict:
    """Benchmark cache lookup (hit) vs full Score rebuild (miss).

    Populates a ScoreCache, then measures:
      - Hit path: fingerprint lookup in cache
      - Miss path: Score reconstruction from scratch
    """
    score = make_score(num_tasks, dependency_style="chain")
    fp = score_fingerprint(score)
    cache = ScoreCache()
    cache.store(score, fingerprint=fp)

    # Benchmark cache hit (lookup only)
    def cache_hit():
        return cache.lookup(fp)

    hit_result = run_benchmark(
        "cache_hit",
        cache_hit,
        params={"num_tasks": num_tasks, "path": "hit"},
    )

    # Benchmark cache miss (full rebuild)
    def cache_miss():
        miss_fp = "nonexistent_fingerprint_for_miss"
        payload = cache.lookup(miss_fp)
        if payload is None:
            # Simulate rebuild on miss
            rebuilt = make_score(num_tasks, dependency_style="chain")
            cache.store(rebuilt)
        return rebuilt

    miss_result = run_benchmark(
        "cache_miss_rebuild",
        cache_miss,
        params={"num_tasks": num_tasks, "path": "miss"},
    )

    # Benchmark cached_score helper (hit path)
    call_count = 0

    def builder():
        nonlocal call_count
        call_count += 1
        return make_score(num_tasks, dependency_style="chain")

    def cached_score_hit():
        return cached_score(cache, builder, fingerprint_hint=fp)

    cached_result = run_benchmark(
        "cached_score_hit",
        cached_score_hit,
        params={"num_tasks": num_tasks},
    )

    # Compute hit rate from a simulated workload
    sim_cache = ScoreCache()
    sim_cache.store(score, fingerprint=fp)
    hits = 0
    misses = 0
    total_lookups = 100
    for i in range(total_lookups):
        # 80% of lookups use the correct fingerprint
        lookup_fp = fp if i % 5 != 0 else f"wrong_{i}"
        if sim_cache.lookup(lookup_fp) is not None:
            hits += 1
        else:
            misses += 1

    hit_rate_pct = hits / total_lookups * 100

    # Return the hit benchmark as the primary result, enriched with extra data
    hit_result["parameters"]["cache_hit_rate_pct"] = hit_rate_pct
    hit_result["cache_miss_rebuild_ms"] = miss_result["wall_time_ms"]
    hit_result["cached_score_hit_ms"] = cached_result["wall_time_ms"]
    hit_result["speedup"] = (
        round(miss_result["wall_time_ms"] / hit_result["wall_time_ms"], 1)
        if hit_result["wall_time_ms"] > 0
        else float("inf")
    )
    return hit_result


def main() -> None:
    print("=" * 90)
    print("Benchmark: score_fingerprint computation and cache performance")
    print("=" * 90)

    results: list[dict] = []

    print("\n--- Fingerprint computation ---")
    for n in TASK_COUNTS:
        r = bench_fingerprint_compute(n)
        results.append(r)
    print_results(results[-len(TASK_COUNTS):])

    print("\n--- Fingerprint with metadata ---")
    for n in TASK_COUNTS:
        r = bench_fingerprint_with_metadata(n)
        results.append(r)
    print_results(results[-len(TASK_COUNTS):])

    print("\n--- Cache hit vs miss ---")
    for n in TASK_COUNTS:
        r = bench_cache_hit_vs_miss(n)
        results.append(r)
        speedup = r.get("speedup", "N/A")
        print(
            f"  num_tasks={n}: hit={r['wall_time_ms']:.4f} ms, "
            f"miss_rebuild={r['cache_miss_rebuild_ms']:.4f} ms, "
            f"speedup={speedup}x, "
            f"simulated_hit_rate={r['parameters']['cache_hit_rate_pct']:.0f}%"
        )

    path = save_results(results, "bench_fingerprint.json")
    print(f"\nResults saved to: {path}")
    print(f"Total benchmarks: {len(results)}")


if __name__ == "__main__":
    main()
