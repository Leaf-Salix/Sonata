#!/usr/bin/env python3
"""Benchmark Score construction at different scales.

Measures time to construct Score objects with varying task counts
(1, 10, 50, 100, 500) and dependency densities (none, chain, dense).

Usage:
    PYTHONPATH=src python benchmarks/bench_score_build.py
"""

import sys
from pathlib import Path

# Ensure the benchmarks package root is importable
sys.path.insert(0, str(Path(__file__).parent))

from _bench_utils import (
    make_score,
    print_results,
    run_benchmark,
    save_results,
)
from sonata import Score


TASK_COUNTS = [1, 10, 50, 100, 500]
DEPENDENCY_STYLES = ["none", "chain", "dense"]
DENSE_DENSITY = 0.05


def bench_score_construction(num_tasks: int, style: str) -> dict:
    """Benchmark Score() constructor for the given parameters."""

    def build():
        return make_score(num_tasks, dependency_style=style, density=DENSE_DENSITY)

    result = run_benchmark(
        f"score_build_{style}",
        build,
        params={"num_tasks": num_tasks, "style": style},
    )
    # Also record the actual dependency count for context
    score = build()
    result["parameters"]["num_dependencies"] = score.dependency_count()
    return result


def bench_score_validate(num_tasks: int) -> dict:
    """Benchmark Score.validate() which checks consistency and cycles."""
    score = make_score(num_tasks, dependency_style="chain")

    def validate():
        return score.validate()

    return run_benchmark(
        "score_validate",
        validate,
        params={"num_tasks": num_tasks},
    )


def main() -> None:
    print("=" * 90)
    print("Benchmark: Score construction at different scales")
    print("=" * 90)

    results: list[dict] = []

    for style in DEPENDENCY_STYLES:
        print(f"\n--- Dependency style: {style} ---")
        for n in TASK_COUNTS:
            r = bench_score_construction(n, style)
            results.append(r)
        print_results([r for r in results if r["parameters"].get("style") == style])

    print(f"\n--- Score validation ---")
    for n in TASK_COUNTS:
        r = bench_score_validate(n)
        results.append(r)
    print_results([r for r in results if r["benchmark"] == "score_validate"])

    path = save_results(results, "bench_score_build.json")
    print(f"\nResults saved to: {path}")
    print(f"Total benchmarks: {len(results)}")


if __name__ == "__main__":
    main()
