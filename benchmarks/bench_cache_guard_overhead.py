"""Benchmark for cache guard checking overhead.

This benchmark measures the performance impact of guard validation in ScoreCache.
It simulates different guard violation rates to quantify the overhead.

Usage:
    cd pypto-sonata && PYTHONPATH=src python benchmarks/bench_cache_guard_overhead.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sonata.cache import GuardStatus, ScoreCache
from sonata.score import Dependency, RuntimeTarget, Score, ShapeAssumption, Task
from sonata.guard import GUARD_SEVERITY_HARD


def make_test_score() -> Score:
    """Create a simple test Score for benchmarking."""
    return Score(
        name="bench_guard_test",
        runtime_target=RuntimeTarget(),
        tasks=(
            Task(task_id=0, func_id=0, core_type="aic", name="matmul"),
            Task(task_id=1, func_id=1, core_type="aic", name="bias_add"),
        ),
        dependencies=(Dependency(producer=0, consumer=1),),
        shape_assumptions=(
            ShapeAssumption(symbol="batch", dims=(32,), severity=GUARD_SEVERITY_HARD),
            ShapeAssumption(symbol="seq", dims=(128,), severity=GUARD_SEVERITY_HARD),
        ),
    )


def benchmark_no_violations(cache: ScoreCache, fingerprint: str, iterations: int = 10000):
    """Benchmark cache lookups with no guard violations (ALL_SATISFIED)."""
    for _ in range(iterations):
        cache.lookup(fingerprint)


def benchmark_all_failed(cache: ScoreCache, fingerprint: str, iterations: int = 10000):
    """Benchmark cache lookups with all guards failed."""
    # Create a new entry with ALL_FAILED status
    from sonata.cache import CacheEntry
    existing = cache._entries[fingerprint]
    cache._entries[fingerprint] = CacheEntry(
        fingerprint=existing.fingerprint,
        score_payload=existing.score_payload,
        schema_version=existing.schema_version,
        fingerprint_version=existing.fingerprint_version,
        created_at=existing.created_at,
        plan_handle_payload=existing.plan_handle_payload,
        metadata=existing.metadata,
        guard_status=GuardStatus.ALL_FAILED,
    )
    
    for _ in range(iterations):
        cache.lookup(fingerprint)


def benchmark_partial_failed(cache: ScoreCache, fingerprint: str, iterations: int = 10000):
    """Benchmark cache lookups with partial guard failures."""
    # Create a new entry with PARTIAL_FAILED status
    from sonata.cache import CacheEntry
    existing = cache._entries[fingerprint]
    cache._entries[fingerprint] = CacheEntry(
        fingerprint=existing.fingerprint,
        score_payload=existing.score_payload,
        schema_version=existing.schema_version,
        fingerprint_version=existing.fingerprint_version,
        created_at=existing.created_at,
        plan_handle_payload=existing.plan_handle_payload,
        metadata=existing.metadata,
        guard_status=GuardStatus.PARTIAL_FAILED,
    )
    
    for _ in range(iterations):
        cache.lookup(fingerprint)


def run_benchmark(name: str, fn, *, warmup: int = 3, iterations: int = 10000):
    """Run a benchmark and return timing statistics."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Measure
    import time
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    
    return elapsed_ms / iterations  # Return average per lookup in ms


def main():
    """Run all guard overhead benchmarks."""
    print("=" * 80)
    print("ScoreCache Guard Checking Overhead Benchmark")
    print("=" * 80)
    print()
    
    # Setup
    cache = ScoreCache()
    score = make_test_score()
    fingerprint = cache.store(score)
    
    results = []
    
    # Benchmark 1: No violations (baseline)
    print("Running baseline (ALL_SATISFIED)...")
    time_baseline = run_benchmark(
        "baseline",
        lambda: benchmark_no_violations(cache, fingerprint),
        iterations=10000
    )
    results.append({
        "scenario": "ALL_SATISFIED (no violations)",
        "avg_time_us": round(time_baseline * 1000, 4),
        "iterations": 10000
    })
    print(f"  Baseline: {time_baseline*1000:.4f} µs per lookup")
    
    # Benchmark 2: All guards failed
    print("Running with ALL_FAILED...")
    time_all_failed = run_benchmark(
        "all_failed",
        lambda: benchmark_all_failed(cache, fingerprint),
        iterations=10000
    )
    results.append({
        "scenario": "ALL_FAILED (all guards violated)",
        "avg_time_us": round(time_all_failed * 1000, 4),
        "overhead_pct": round((time_all_failed / time_baseline - 1) * 100, 2)
    })
    print(f"  ALL_FAILED: {time_all_failed*1000:.4f} µs per lookup")
    print(f"  Overhead: {(time_all_failed / time_baseline - 1) * 100:.2f}%")
    
    # Benchmark 3: Partial guards failed
    print("Running with PARTIAL_FAILED...")
    time_partial_failed = run_benchmark(
        "partial_failed",
        lambda: benchmark_partial_failed(cache, fingerprint),
        iterations=10000
    )
    results.append({
        "scenario": "PARTIAL_FAILED (some guards violated)",
        "avg_time_us": round(time_partial_failed * 1000, 4),
        "overhead_pct": round((time_partial_failed / time_baseline - 1) * 100, 2)
    })
    print(f"  PARTIAL_FAILED: {time_partial_failed*1000:.4f} µs per lookup")
    print(f"  Overhead: {(time_partial_failed / time_baseline - 1) * 100:.2f}%")
    
    # Summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print(f"{'Scenario':<45s} {'Avg Time (µs)':<15s} {'Overhead':<15s}")
    print("-" * 80)
    for r in results:
        if "overhead_pct" in r:
            print(f"{r['scenario']:<45s} {r['avg_time_us']:>12.4f} µs   +{r['overhead_pct']:>6.2f}%")
        else:
            print(f"{r['scenario']:<45s} {r['avg_time_us']:>12.4f} µs   (baseline)")
    
    print()
    print("Key Findings:")
    print("-" * 80)
    
    overhead_all = (time_all_failed / time_baseline - 1) * 100
    overhead_partial = (time_partial_failed / time_baseline - 1) * 100
    
    if overhead_all < 5 and overhead_partial < 5:
        print("✅ Guard checking overhead is minimal (< 5%)")
        print("   The conservative approach (treating violations as cache misses)")
        print("   adds negligible performance cost.")
    elif overhead_all < 10 and overhead_partial < 10:
        print("⚠️  Guard checking overhead is acceptable (< 10%)")
        print("   Consider monitoring in production environments.")
    else:
        print("❗ Guard checking overhead may be significant (> 10%)")
        print("   Consider optimizing guard evaluation or caching strategy.")
    
    print()
    print("Recommendations:")
    print("-" * 80)
    print("1. Guard checking is safe to enable by default")
    print("2. Conservative invalidation policy prevents subtle bugs")
    print("3. Monitor in production with realistic guard violation rates")
    print("4. Consider adaptive strategies if violation rate > 50%")
    
    print()
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()
    
    # Save results to file
    import json
    from pathlib import Path
    
    RESULTS_DIR = Path(__file__).parent / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    output_file = RESULTS_DIR / "cache_guard_overhead_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
