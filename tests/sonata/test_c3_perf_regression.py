"""C3: Performance regression — Sonata overhead vs no-Sonata baseline.

Measures end-to-end time for a representative ST test (test_abs.py, 4 subtests)
and compares wall-clock time between:

1. Baseline: run without ``--with-sonata``
2. Sonata:  run with ``--with-sonata`` (analysis + schedule + config injection)

Budget: Sonata overhead < 5% of baseline time.

Uses subprocess with ``--rootdir`` to ensure correct conftest discovery.
"""

import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJ_ROOT = Path(__file__).resolve().parents[2]
ST_TEST = PROJ_ROOT / "upstream" / "pypto" / "tests" / "st" / "runtime" / "ops" / "test_abs.py"
UPSTREAM_PYPTO = PROJ_ROOT / "upstream" / "pypto"
VENV_PYTHON = Path(sys.executable)
REPORT_DIR = PROJ_ROOT / "reports" / "detail" / "review"
PERF_BUDGET_PCT = 5.0  # allowed overhead percentage

SAMPLES = 3

# a2a3sim tests need PTOAS cross-compiler (absent in CI, present on dev machine)
_HAS_A2A3SIM = shutil.which("ptoas") is not None or bool(os.environ.get("PTOAS_ROOT"))
_skip_no_a2a3sim = pytest.mark.skipif(
    not _HAS_A2A3SIM,
    reason="a2a3sim not available (PTOAS not found)",
)


def _run_test(with_sonata: bool, sample: int) -> float:
    """Run test_abs.py once, return wall-clock seconds."""
    cmd = [
        str(VENV_PYTHON),
        "-m", "pytest",
        "--platform=a2a3sim",
        "--no-header", "-q", "-p", "no:cacheprovider",
    ]
    if with_sonata:
        cmd.extend(["--with-sonata", "--rootdir", str(PROJ_ROOT), str(ST_TEST)])
    else:
        cmd.extend(["--rootdir", str(UPSTREAM_PYPTO), "tests/st/runtime/ops/test_abs.py"])

    env = os.environ.copy()
    env.update({
        "PTOAS_ROOT": "/Users/jiayetcs/Desktop/Project/PyPTO/ptoas",
        "PTO_ISA_ROOT": "/Users/jiayetcs/Desktop/Project/PyPTO/ptoisa",
    })
    cwd = str(UPSTREAM_PYPTO if not with_sonata else PROJ_ROOT)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env, cwd=cwd)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        pytest.fail(
            f"{'Sonata' if with_sonata else 'Baseline'} test run {sample} "
            f"FAILED (rc={result.returncode}).\n"
            f"stdout last 200: {result.stdout[-200:]}\n"
            f"stderr: {result.stderr[:200]}"
        )

    return elapsed


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms"


def run_c3_benchmark() -> dict:
    """Run C3 benchmark and return results.

    Samples are interleaved (baseline, sonata, baseline, sonata, ...) rather
    than run as two separate blocks. Each subprocess invocation pays for a
    full ptoas+g++ compile, which heats up the machine; if all baseline
    samples ran first, system-wide drift (thermal throttling, background
    load) over the run would land entirely on whichever mode runs second,
    masquerading as "Sonata overhead" instead of as noise distributed
    between both modes.
    """
    print(f"\n=== C3: Performance Regression ({SAMPLES} samples per mode, interleaved) ===\n")
    print(f"Test: {ST_TEST}")
    print(f"Budget: Sonata overhead < {PERF_BUDGET_PCT}%\n")

    baseline_times = []
    sonata_times = []

    for s in range(1, SAMPLES + 1):
        t = _run_test(with_sonata=False, sample=s)
        baseline_times.append(t)
        print(f"  baseline sample {s}: {_format_ms(t)}")

        t = _run_test(with_sonata=True, sample=s)
        sonata_times.append(t)
        print(f"  sonata   sample {s}: {_format_ms(t)}")

    # Median, not mean: each sample is a full subprocess (ptoas + g++ compile),
    # so a single sample disrupted by unrelated system load (e.g. Spotlight,
    # background indexing) easily swings 2x: a mean would let that one outlier
    # dominate a 3-sample budget check, a median absorbs it.
    baseline_avg = statistics.median(baseline_times)
    sonata_avg = statistics.median(sonata_times)
    overhead_pct = (sonata_avg - baseline_avg) / baseline_avg * 100

    print(f"\n  baseline median: {_format_ms(baseline_avg)}")
    print(f"  sonata   median: {_format_ms(sonata_avg)}")
    print(f"  overhead:       {overhead_pct:+.1f}%")

    return {
        "baseline_samples": baseline_times,
        "baseline_avg_s": baseline_avg,
        "sonata_samples": sonata_times,
        "sonata_avg_s": sonata_avg,
        "overhead_pct": overhead_pct,
        "within_budget": overhead_pct <= PERF_BUDGET_PCT,
    }


@_skip_no_a2a3sim
def test_c3_performance_regression():
    """Assert Sonata overhead < 5% of baseline."""
    results = run_c3_benchmark()
    assert results["within_budget"], (
        f"Sonata overhead {results['overhead_pct']:.1f}% exceeds budget of {PERF_BUDGET_PCT}%"
    )
    print(f"\n✅ Sonata overhead {results['overhead_pct']:.1f}% within budget ({PERF_BUDGET_PCT}%)")


def save_report(results: dict) -> None:
    """Save results to reports/detail/review/."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "v0.27-perf-results.md"
    lines = [
        "# v0.27 Performance Regression Results\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Test: `test_abs.py` (4 subtests)\n",
        f"Platform: a2a3sim (simulator)\n",
        f"Budget: Sonata overhead < {PERF_BUDGET_PCT}%\n",
        "",
        "## Results\n",
        "| Mode | Samples | Avg (s) | vs Baseline |",
        "|------|---------|---------|-------------|",
        f"| Baseline | {len(results['baseline_samples'])} | {results['baseline_avg_s']:.2f} | — |",
        f"| Sonata | {len(results['sonata_samples'])} | {results['sonata_avg_s']:.2f} | {results['overhead_pct']:+.1f}% |",
        "",
    ]
    if results["within_budget"]:
        lines.append(f"**Result**: ✅ Within budget ({results['overhead_pct']:.1f}% < {PERF_BUDGET_PCT}%).\n")
    else:
        lines.append(f"**Result**: ❌ Exceeds budget ({results['overhead_pct']:.1f}% >= {PERF_BUDGET_PCT}%).\n")

    lines.extend([
        f"## Sample Details\n",
        f"| Run | Baseline (s) | Sonata (s) |",
        f"|-----|-------------|------------|",
    ])
    for i in range(max(len(results["baseline_samples"]), len(results["sonata_samples"]))):
        b = f"{results['baseline_samples'][i]:.2f}" if i < len(results["baseline_samples"]) else "—"
        s = f"{results['sonata_samples'][i]:.2f}" if i < len(results["sonata_samples"]) else "—"
        lines.append(f"| {i+1} | {b} | {s} |")

    report_path.write_text("\n".join(lines))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    results = run_c3_benchmark()
    save_report(results)
    if not results["within_budget"]:
        sys.exit(1)
