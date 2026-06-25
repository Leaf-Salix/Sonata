"""Fail-open tests for sonata_hook.h (B3 + B4).

Runs the C++ test_hook_harness gtest binary and validates all 6 fail-open
modes:

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
All sub-tests share the cached gtest JSON results via the ``harness_output``
fixture -- no 8x redundant subprocess invocations.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).parent / "tools"
HARNESS_BIN = TOOLS_DIR / "test_hook_harness"


@pytest.fixture(scope="module")
def harness_output():
    """Compile and run the C++ gtest harness once; cache JSON results for all tests."""
    result = subprocess.run(
        ["make", "-C", str(TOOLS_DIR), "all"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.skip(f"Failed to compile test_hook_harness: {result.stderr}")

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "results.json"
        result = subprocess.run(
            [str(HARNESS_BIN), f"--gtest_output=json:{json_path}"],
            capture_output=True, text=True, timeout=30
        )
        if result.stderr:
            print("STDERR:", result.stderr)
        results = json.loads(json_path.read_text()) if json_path.exists() else None
    result.results = results
    return result


def _find_test(results, name, suite="SonataHook"):
    """Return the gtest testsuite entry for ``suite.name``, or None if absent."""
    if results is None:
        return None
    for testsuite in results["testsuites"]:
        if testsuite["name"] != suite:
            continue
        for test in testsuite["testsuite"]:
            if test["name"] == name:
                return test
    return None


def _assert_passed(results, name, suite="SonataHook"):
    """Assert that gtest case ``suite.name`` ran and reported no failures."""
    test = _find_test(results, name, suite)
    assert test is not None, f"gtest case {suite}.{name} not found in results"
    assert "failures" not in test, (
        f"gtest case {suite}.{name} failed: {test.get('failures')}"
    )


def test_hook_harness_passes(harness_output):
    """Run the full C++ fail-open test harness (B3 + B4 modes 1-4)."""
    print(harness_output.stdout)

    assert harness_output.returncode == 0, (
        f"test_hook_harness failed (rc={harness_output.returncode}):\n{harness_output.stdout}"
    )

    results = harness_output.results
    assert results is not None, "gtest did not produce a JSON results file"
    assert results["failures"] == 0, f"{results['failures']} gtest case(s) failed"
    assert results["tests"] > 0, "no gtest cases ran"


def test_hook_harness_b3_disabled_mode(harness_output):
    """B3: Verify SONATA_HOOK_DISABLED is returned when SONATA_ENABLED is not set."""
    _assert_passed(harness_output.results, "DisabledModeSkipsAicpu")


def test_hook_harness_b4_null_blob(harness_output):
    """B4 mode 1: null blob -> SONATA_HOOK_ERROR, aicpu_entry not called."""
    _assert_passed(harness_output.results, "NullBlobReturnsError")


def test_hook_harness_b4_wrong_magic(harness_output):
    """B4 mode 2: wrong magic number -> SONATA_HOOK_ERROR."""
    _assert_passed(harness_output.results, "WrongMagicReturnsError")


def test_hook_harness_b4_wrong_version(harness_output):
    """B4 mode 3: wrong version -> SONATA_HOOK_ERROR."""
    _assert_passed(harness_output.results, "WrongVersionReturnsError")


def test_hook_harness_b4_truncated_blob(harness_output):
    """B4 mode 4: truncated / bounds-violating blob -> SONATA_HOOK_ERROR."""
    _assert_passed(harness_output.results, "TruncatedBlobReturnsError")
    _assert_passed(harness_output.results, "NegativeRegionsReturnsError")


def test_hook_struct_sizes(harness_output):
    """Verify packed struct sizes match flat_schedule.h expectations."""
    _assert_passed(harness_output.results, "StructSizesArePacked")


def test_cross_language_binary_validation():
    """Python-produced .bin validated by C sonata_hook in cross-language round-trip."""
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
        assert "OK: process_schedule OK" in result.stdout, (
            f"Hook did not accept Python-produced .bin:\n{result.stdout}"
        )
