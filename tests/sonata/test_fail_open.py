"""Fail-open tests for sonata_hook.h (B3 + B4).

Runs the C++ test_hook_harness binary and validates all 6 fail-open modes:

B3: no SONATA_ENABLED -> original path (SONATA_HOOK_DISABLED)
B4:
  1. schedule blob missing / null -> SONATA_HOOK_ERROR
  2. wrong magic number -> SONATA_HOOK_ERROR
  3. wrong version -> SONATA_HOOK_ERROR
  4. mismatched task count / truncated -> SONATA_HOOK_ERROR
  5. guard condition false -> SONATA_HOOK_SKIP (Python-level; tested in test_guard.py)
  6. backend rejects schedule -> SONATA_HOOK_ERROR (Python-level; tested in test_pipeline.py)

The C++ harness tests modes 1-4 and B3 directly against the hook API.
Modes 5-6 are covered by existing Python guard/pipeline tests.
"""

import os
import subprocess
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).parent / "tools"
HARNESS_BIN = TOOLS_DIR / "test_hook_harness"


@pytest.fixture(scope="module")
def harness_binary():
    """Compile the C++ harness if not already built."""
    if HARNESS_BIN.exists():
        return str(HARNESS_BIN)

    # Build the harness
    result = subprocess.run(
        ["make", "-C", str(TOOLS_DIR), "-f", "Makefile", "all"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.skip(f"Failed to compile test_hook_harness: {result.stderr}")
    return str(HARNESS_BIN)


def test_hook_harness_passes(harness_binary):
    """Run the full C++ fail-open test harness (B3 + B4 modes 1-4)."""
    result = subprocess.run(
        [harness_binary],
        capture_output=True, text=True, timeout=30
    )

    # Print harness output for debugging
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    assert result.returncode == 0, (
        f"test_hook_harness failed (rc={result.returncode}):\n{result.stdout}"
    )

    # Verify all expected tests ran
    assert "=== Results:" in result.stdout
    # Parse "37/37 passed, 0 failed"
    for line in result.stdout.splitlines():
        if "Results:" in line and "passed" in line:
            parts = line.split()
            passed_total = parts[2]  # "37/37"
            passed, total = passed_total.split("/")
            failed = parts[4].rstrip(",")
            assert int(failed) == 0, f"{failed} tests failed"
            assert int(passed) == int(total), f"Only {passed}/{total} passed"
            break
    else:
        pytest.fail("Could not parse test_hook_harness results")


def test_hook_harness_b3_disabled_mode(harness_binary):
    """B3: Verify SONATA_HOOK_DISABLED is returned when SONATA_ENABLED is not set."""
    result = subprocess.run(
        [harness_binary],
        capture_output=True, text=True, timeout=30
    )
    assert "PASS: process_schedule returns DISABLED (no SONATA_ENABLED)" in result.stdout
    assert "PASS: aicpu_entry NOT called when disabled" in result.stdout


def test_hook_harness_b4_null_blob(harness_binary):
    """B4 mode 1: null blob -> SONATA_HOOK_ERROR, aicpu_entry not called."""
    result = subprocess.run(
        [harness_binary],
        capture_output=True, text=True, timeout=30
    )
    assert "PASS: null blob returns ERROR" in result.stdout
    assert "PASS: aicpu_entry NOT called for null blob" in result.stdout


def test_hook_harness_b4_wrong_magic(harness_binary):
    """B4 mode 2: wrong magic number -> SONATA_HOOK_ERROR."""
    result = subprocess.run(
        [harness_binary],
        capture_output=True, text=True, timeout=30
    )
    assert "PASS: wrong magic returns ERROR" in result.stdout
    assert "PASS: aicpu_entry NOT called for wrong magic" in result.stdout


def test_hook_harness_b4_wrong_version(harness_binary):
    """B4 mode 3: wrong version -> SONATA_HOOK_ERROR."""
    result = subprocess.run(
        [harness_binary],
        capture_output=True, text=True, timeout=30
    )
    assert "PASS: wrong version returns ERROR" in result.stdout
    assert "PASS: aicpu_entry NOT called for wrong version" in result.stdout


def test_hook_harness_b4_truncated_blob(harness_binary):
    """B4 mode 4: truncated / bounds-violating blob -> SONATA_HOOK_ERROR."""
    result = subprocess.run(
        [harness_binary],
        capture_output=True, text=True, timeout=30
    )
    assert "PASS: truncated blob returns ERROR" in result.stdout
    assert "PASS: aicpu_entry NOT called for truncated blob" in result.stdout
    assert "PASS: negative regions returns ERROR" in result.stdout


def test_hook_struct_sizes(harness_binary):
    """Verify packed struct sizes match flat_schedule.h expectations."""
    result = subprocess.run(
        [harness_binary],
        capture_output=True, text=True, timeout=30
    )
    assert "PASS: FlatSchedule is 88 bytes (packed)" in result.stdout
    assert "PASS: FlatRegion is 24 bytes (packed)" in result.stdout
    assert "PASS: FlatTask is 16 bytes (packed)" in result.stdout
    assert "PASS: FlatArg is 6 bytes (packed)" in result.stdout
    assert "PASS: FlatDep is 8 bytes (packed)" in result.stdout
