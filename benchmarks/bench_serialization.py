#!/usr/bin/env python3
"""Benchmark serialization and deserialization round-trips.

Measures:
  - score_to_dict() for different Score sizes
  - score_to_json() for different Score sizes
  - score_from_dict() for different Score sizes
  - score_from_json() for different Score sizes
  - Full round-trip (serialize then deserialize)

Usage:
    PYTHONPATH=src python benchmarks/bench_serialization.py
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
from sonata import (
    score_fingerprint,
    score_from_dict,
    score_from_json,
    score_to_dict,
    score_to_json,
)


TASK_COUNTS = [1, 10, 50, 100, 500]


def bench_score_to_dict(num_tasks: int) -> dict:
    """Benchmark score_to_dict() serialization."""
    score = make_score(num_tasks, dependency_style="chain")

    def serialize():
        return score_to_dict(score)

    return run_benchmark(
        "score_to_dict",
        serialize,
        params={"num_tasks": num_tasks},
    )


def bench_score_to_json(num_tasks: int) -> dict:
    """Benchmark score_to_json() serialization (includes JSON encoding)."""
    score = make_score(num_tasks, dependency_style="chain")

    def serialize():
        return score_to_json(score, indent=None)

    return run_benchmark(
        "score_to_json",
        serialize,
        params={"num_tasks": num_tasks},
    )


def bench_score_from_dict(num_tasks: int) -> dict:
    """Benchmark score_from_dict() deserialization."""
    score = make_score(num_tasks, dependency_style="chain")
    payload = score_to_dict(score)

    def deserialize():
        return score_from_dict(payload)

    return run_benchmark(
        "score_from_dict",
        deserialize,
        params={"num_tasks": num_tasks},
    )


def bench_score_from_json(num_tasks: int) -> dict:
    """Benchmark score_from_json() deserialization (includes JSON decoding)."""
    score = make_score(num_tasks, dependency_style="chain")
    json_str = score_to_json(score, indent=None)

    def deserialize():
        return score_from_json(json_str)

    return run_benchmark(
        "score_from_json",
        deserialize,
        params={"num_tasks": num_tasks},
    )


def bench_round_trip(num_tasks: int) -> dict:
    """Benchmark full serialize + deserialize round-trip."""
    score = make_score(num_tasks, dependency_style="chain")

    def round_trip():
        payload = score_to_dict(score)
        restored = score_from_dict(payload)
        return restored

    result = run_benchmark(
        "round_trip_dict",
        round_trip,
        params={"num_tasks": num_tasks},
    )

    # Also measure JSON round-trip
    def json_round_trip():
        json_str = score_to_json(score, indent=None)
        return score_from_json(json_str)

    json_result = run_benchmark(
        "round_trip_json",
        json_round_trip,
        params={"num_tasks": num_tasks},
    )

    result["json_round_trip_ms"] = json_result["wall_time_ms"]

    # Verify correctness
    restored = score_from_dict(score_to_dict(score))
    fp_orig = score_fingerprint(score)
    fp_restored = score_fingerprint(restored)
    result["parameters"]["fingerprint_match"] = fp_orig == fp_restored

    return result


def bench_payload_size(num_tasks: int) -> dict:
    """Measure serialized payload size for different Score sizes."""
    score = make_score(num_tasks, dependency_style="chain")
    payload = score_to_dict(score)
    json_str = score_to_json(score, indent=None)

    import json
    dict_size = len(json.dumps(payload))
    json_size = len(json_str)

    return {
        "benchmark": "payload_size",
        "parameters": {
            "num_tasks": num_tasks,
            "num_dependencies": score.dependency_count(),
            "dict_bytes": dict_size,
            "json_bytes": json_size,
        },
        "wall_time_ms": 0.0,
        "median_ms": 0.0,
        "min_ms": 0.0,
        "max_ms": 0.0,
        "iterations": 0,
        "throughput": 0.0,
    }


def main() -> None:
    print("=" * 90)
    print("Benchmark: serialization / deserialization round-trips")
    print("=" * 90)

    results: list[dict] = []

    print("\n--- score_to_dict ---")
    for n in TASK_COUNTS:
        results.append(bench_score_to_dict(n))
    print_results(results[-len(TASK_COUNTS):])

    print("\n--- score_to_json ---")
    for n in TASK_COUNTS:
        results.append(bench_score_to_json(n))
    print_results(results[-len(TASK_COUNTS):])

    print("\n--- score_from_dict ---")
    for n in TASK_COUNTS:
        results.append(bench_score_from_dict(n))
    print_results(results[-len(TASK_COUNTS):])

    print("\n--- score_from_json ---")
    for n in TASK_COUNTS:
        results.append(bench_score_from_json(n))
    print_results(results[-len(TASK_COUNTS):])

    print("\n--- Round-trip (dict + json) ---")
    for n in TASK_COUNTS:
        r = bench_round_trip(n)
        results.append(r)
        print(
            f"  num_tasks={n}: dict_round_trip={r['wall_time_ms']:.4f} ms, "
            f"json_round_trip={r['json_round_trip_ms']:.4f} ms, "
            f"fingerprint_match={r['parameters']['fingerprint_match']}"
        )

    print("\n--- Payload sizes ---")
    for n in TASK_COUNTS:
        r = bench_payload_size(n)
        results.append(r)
        p = r["parameters"]
        print(
            f"  num_tasks={n}: dict={p['dict_bytes']} bytes, "
            f"json={p['json_bytes']} bytes, "
            f"deps={p['num_dependencies']}"
        )

    path = save_results(results, "bench_serialization.json")
    print(f"\nResults saved to: {path}")
    print(f"Total benchmarks: {len(results)}")


if __name__ == "__main__":
    main()
