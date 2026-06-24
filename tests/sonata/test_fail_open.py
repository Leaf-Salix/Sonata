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

NOTE: the C++ binary is compiled once and run ONCE per test-module scope.
All sub-tests share the cached stdout via ``harness_output`` fixture — no
8x redundant subprocess invocations.
"""

import os
import subprocess
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).parent / "tools"
HARNESS_BIN = TOOLS_DIR / "test_hook_harness"


@pytest.fixture(scope="module")
def harness_output():
    """Compile and run the C++ harness once; cache stdout for all tests."""
    if not HARNESS_BIN.exists():
        result = subprocess.run(
            ["make", "-C", str(TOOLS_DIR), "all"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            pytest.skip(f"Failed to compile test_hook_harness: {result.stderr}")

    result = subprocess.run(
        [str(HARNESS_BIN)],
        capture_output=True, text=True, timeout=30
    )
    if result.stderr:
        print("STDERR:", result.stderr)
    return result


def test_hook_harness_passes(harness_output):
    """Run the full C++ fail-open test harness (B3 + B4 modes 1-4)."""
    stdout = harness_output.stdout
    print(stdout)

    assert harness_output.returncode == 0, (
        f"test_hook_harness failed (rc={harness_output.returncode}):\n{stdout}"
    )

    # Verify all expected tests ran
    assert "=== Results:" in stdout
    for line in stdout.splitlines():
        if "Results:" in line and "passed" in line:
            parts = line.split()
            passed_total = parts[2]  # e.g. "37/37"
            passed, total = passed_total.split("/")
            failed = parts[4].rstrip(",")
            assert int(failed) == 0, f"{failed} tests failed"
            assert int(passed) == int(total), f"Only {passed}/{total} passed"
            return
    pytest.fail("Could not parse test_hook_harness results")


def test_hook_harness_b3_disabled_mode(harness_output):
    """B3: Verify SONATA_HOOK_DISABLED is returned when SONATA_ENABLED is not set."""
    assert "PASS: process_schedule returns DISABLED (no SONATA_ENABLED)" in harness_output.stdout
    assert "PASS: aicpu_entry NOT called when disabled" in harness_output.stdout


def test_hook_harness_b4_null_blob(harness_output):
    """B4 mode 1: null blob -> SONATA_HOOK_ERROR, aicpu_entry not called."""
    assert "PASS: null blob returns ERROR" in harness_output.stdout
    assert "PASS: aicpu_entry NOT called for null blob" in harness_output.stdout


def test_hook_harness_b4_wrong_magic(harness_output):
    """B4 mode 2: wrong magic number -> SONATA_HOOK_ERROR."""
    assert "PASS: wrong magic returns ERROR" in harness_output.stdout
    assert "PASS: aicpu_entry NOT called for wrong magic" in harness_output.stdout


def test_hook_harness_b4_wrong_version(harness_output):
    """B4 mode 3: wrong version -> SONATA_HOOK_ERROR."""
    assert "PASS: wrong version returns ERROR" in harness_output.stdout
    assert "PASS: aicpu_entry NOT called for wrong version" in harness_output.stdout


def test_hook_harness_b4_truncated_blob(harness_output):
    """B4 mode 4: truncated / bounds-violating blob -> SONATA_HOOK_ERROR."""
    assert "PASS: truncated blob returns ERROR" in harness_output.stdout
    assert "PASS: aicpu_entry NOT called for truncated blob" in harness_output.stdout
    assert "PASS: negative regions returns ERROR" in harness_output.stdout


def test_hook_struct_sizes(harness_output):
    """Verify packed struct sizes match flat_schedule.h expectations."""
    assert "PASS: FlatSchedule is 88 bytes (packed)" in harness_output.stdout
    assert "PASS: FlatRegion is 24 bytes (packed)" in harness_output.stdout
    assert "PASS: FlatTask is 16 bytes (packed)" in harness_output.stdout
    assert "PASS: FlatArg is 6 bytes (packed)" in harness_output.stdout
    assert "PASS: FlatDep is 8 bytes (packed)" in harness_output.stdout


def test_cross_language_binary_validation():
    """Python-produced .bin validated by C sonata_hook in cross-language round-trip."""
    import subprocess
    import tempfile
    from pathlib import Path
    from sonata.schedule import SonataScheduleContract

    c = SonataScheduleContract(fingerprint="cross_lang")
    data = c.to_binary()

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "test.bin"
        bin_path.write_bytes(data)

        result = subprocess.run(
            [str(HARNESS_BIN), str(bin_path)],
            capture_output=True, text=True, timeout=30
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        assert result.returncode == 0, (
            f"Cross-language validation failed (rc={result.returncode}):\n{result.stdout}"
        )
        assert "PASS: process_schedule OK" in result.stdout, (
            f"Hook did not accept Python-produced .bin:\n{result.stdout}"
        )
