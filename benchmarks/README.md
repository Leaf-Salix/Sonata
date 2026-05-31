# Sonata Benchmarks

Performance benchmarks for the Sonata static scheduling compiler layer.

## Overview

These benchmarks measure the performance of core Sonata operations using
synthetic workloads. They do **not** depend on PyPTO C++ bindings -- all
benchmarks use pure Python mock data.

| Benchmark | File | What it measures |
|-----------|------|------------------|
| Score construction | `bench_score_build.py` | Time to build `Score` objects at different scales (1-500 tasks) with varying dependency densities (none, chain, dense). Also measures `Score.validate()` cost. |
| Fingerprint | `bench_fingerprint.py` | `score_fingerprint()` computation time for different Score sizes. Cache hit vs miss performance and simulated hit rates using `ScoreCache`. |
| Serialization | `bench_serialization.py` | `score_to_dict`, `score_to_json`, `score_from_dict`, `score_from_json` round-trip times and payload sizes for different Score sizes. |
| Eligibility | `bench_eligibility.py` | `check_static_eligibility` with mock IR structures of varying complexity (1-500 calls, nested functions, eligible and ineligible graphs). |

## Running benchmarks

From the `pypto-sonata/` directory:

```bash
# Run a single benchmark
PYTHONPATH=src python benchmarks/bench_score_build.py

# Run all benchmarks
PYTHONPATH=src python benchmarks/bench_score_build.py
PYTHONPATH=src python benchmarks/bench_fingerprint.py
PYTHONPATH=src python benchmarks/bench_serialization.py
PYTHONPATH=src python benchmarks/bench_eligibility.py
```

Each benchmark:

1. Prints structured timing results to stdout.
2. Saves results as JSON to `benchmarks/results/`.

## Results

Results are saved as JSON files under `benchmarks/results/`:

- `bench_score_build.json` -- Score construction timings
- `bench_fingerprint.json` -- Fingerprint and cache timings
- `bench_serialization.json` -- Serialization round-trip timings
- `bench_eligibility.json` -- Eligibility check timings
- `baseline.json` -- Consolidated baseline from all benchmarks

Each entry contains:

| Field | Description |
|-------|-------------|
| `benchmark` | Benchmark name |
| `parameters` | Input parameters (task count, style, etc.) |
| `wall_time_ms` | Mean wall-clock time across measured iterations |
| `median_ms` | Median wall-clock time |
| `min_ms` | Fastest iteration |
| `max_ms` | Slowest iteration |
| `iterations` | Number of measured iterations |
| `throughput` | Operations per second (1000 / mean_ms) |

## Configuration

Each benchmark script uses:

- **Warmup iterations**: 3 (not measured, primes caches and JIT)
- **Measured iterations**: 10
- **Timing**: `time.perf_counter()` for wall-clock accuracy

## Generating a baseline

To regenerate `baseline.json`, run all four benchmark scripts sequentially.
Then consolidate with:

```bash
PYTHONPATH=src python -c "
import json, glob
results = []
for f in sorted(glob.glob('benchmarks/results/bench_*.json')):
    results.extend(json.loads(open(f).read()))
json.dump(results, open('benchmarks/results/baseline.json', 'w'), indent=2)
print(f'Baseline: {len(results)} entries')
"
```
