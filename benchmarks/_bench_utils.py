"""Shared helpers for Sonata benchmarks."""

import json
import time
from pathlib import Path
from typing import Any

from sonata import (
    Dependency,
    RuntimeTarget,
    Score,
    ShapeAssumption,
    Task,
)

RESULTS_DIR = Path(__file__).parent / "results"


def make_task(task_id: int, *, func_id: int = 0, core_type: str = "aic",
              num_args: int = 2) -> Task:
    """Create a single Task with deterministic data."""
    return Task(
        task_id=task_id,
        func_id=func_id,
        core_type=core_type,
        args=tuple(f"arg_{task_id}_{i}" for i in range(num_args)),
        arg_directions=tuple("in" if i == 0 else "out" for i in range(num_args)),
        arg_storage_keys=tuple(f"storage:{task_id}:{i}" for i in range(num_args)),
        name=f"func_{func_id}",
    )


def make_chain_dependencies(num_tasks: int) -> tuple[Dependency, ...]:
    """Create a linear chain: task 0 -> task 1 -> ... -> task N-1."""
    return tuple(
        Dependency(producer=i, consumer=i + 1, kind="data")
        for i in range(num_tasks - 1)
    )


def make_dense_dependencies(num_tasks: int, *, density: float = 0.1) -> tuple[Dependency, ...]:
    """Create dependencies with given density (fraction of all possible edges).

    Only producer < consumer edges are created to keep the graph acyclic.
    """
    deps: list[Dependency] = []
    for producer in range(num_tasks):
        for consumer in range(producer + 1, num_tasks):
            # Deterministic pseudo-random: use modular hash
            if ((producer * 2654435761 + consumer * 40503) % 1000) < int(density * 1000):
                kind = "data" if (producer + consumer) % 3 != 0 else "storage"
                deps.append(Dependency(producer=producer, consumer=consumer, kind=kind))
    return tuple(deps)


def make_score(num_tasks: int, *, name: str | None = None,
               dependency_style: str = "chain",
               density: float = 0.1) -> Score:
    """Build a Score with the given number of tasks and dependency pattern."""
    tasks = tuple(make_task(i, func_id=i % 10) for i in range(num_tasks))

    if dependency_style == "chain":
        deps = make_chain_dependencies(num_tasks)
    elif dependency_style == "dense":
        deps = make_dense_dependencies(num_tasks, density=density)
    elif dependency_style == "none":
        deps = ()
    else:
        raise ValueError(f"unknown dependency_style: {dependency_style}")

    shapes = (
        ShapeAssumption(symbol="batch", dims=(32, 128)),
        ShapeAssumption(symbol="seq", dims=(512,)),
    )

    return Score(
        name=name or f"bench_score_{num_tasks}",
        runtime_target=RuntimeTarget(
            runtime="host_build_graph",
            function_name=f"build_bench_{num_tasks}_graph",
            aicpu_thread_num=4,
        ),
        tasks=tasks,
        dependencies=deps,
        shape_assumptions=shapes,
        metadata={"source": "benchmark", "num_tasks": num_tasks},
    )


def timed(fn, *args, **kwargs) -> tuple[Any, float]:
    """Call fn and return (result, wall_time_ms)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def run_benchmark(name: str, fn, *, warmup: int = 3, iterations: int = 10,
                  params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a benchmark function with warmup and measured iterations.

    Returns a result dict with timing statistics.
    """
    # Warmup
    for _ in range(warmup):
        fn()

    # Measured runs
    times_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed)

    times_ms.sort()
    mean_ms = sum(times_ms) / len(times_ms)
    median_ms = times_ms[len(times_ms) // 2]
    min_ms = times_ms[0]
    max_ms = times_ms[-1]

    return {
        "benchmark": name,
        "parameters": params or {},
        "wall_time_ms": round(mean_ms, 4),
        "median_ms": round(median_ms, 4),
        "min_ms": round(min_ms, 4),
        "max_ms": round(max_ms, 4),
        "iterations": iterations,
        "throughput": round(1000.0 / mean_ms, 2) if mean_ms > 0 else float("inf"),
    }


def save_results(results: list[dict[str, Any]], filename: str) -> Path:
    """Save benchmark results to JSON in the results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    existing: list[dict[str, Any]] = []
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.extend(results)
    path.write_text(
        json.dumps(existing, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    return path


def print_results(results: list[dict[str, Any]]) -> None:
    """Print benchmark results in a structured table format."""
    for r in results:
        params_str = ", ".join(f"{k}={v}" for k, v in r["parameters"].items())
        print(
            f"  {r['benchmark']:40s}  [{params_str:30s}]  "
            f"mean={r['wall_time_ms']:10.4f} ms  "
            f"median={r['median_ms']:10.4f} ms  "
            f"throughput={r['throughput']:12.2f} ops/s"
        )
